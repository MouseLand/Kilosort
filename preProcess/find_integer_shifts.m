function [imin, W0, Ns] = find_integer_shifts(Params, Whs,Ws,mus, ns, iC, Nchan, Nfilt)

nBatches = size(Whs, 2);
nPCs = size(Ws, 1);
% NchanNear = size(Ws, 2);

ibatch0 = min(300, floor(nBatches/2));
% ibatch0 = 1305;

iChan = 1:Nchan;
iUp = mod(iChan + 2-1, Nchan)+1;
iDown = mod(iChan - 2-1, Nchan)+1;
iMap = [iUp(iUp); iUp; iChan; iDown; iDown(iDown)];

% for every batch, compute in parallel its dissimilarity to ALL other batches
Wh0 = single(Whs(:, ibatch0)); % this one is the primary batch
mu0  = mus(:, ibatch0);
W0 = gpuArray.zeros(nPCs , Nchan, Nfilt, 'single');
for t = 1:Nfilt
    W0(:, iC(:, Wh0(t)), t) = Ws(:, :, t, ibatch0);
end

for iter = 1:5
    cc0 = gpuArray.zeros(5, nBatches);
    iclustall = gpuArray.zeros(Nfilt, nBatches, 5);
    for k = 1:5
        % embed the templates from the primary batch back into a full, sparse representation
        W = gpuArray.zeros(nPCs , Nchan, Nfilt, 'single');
        for t = 1:Nfilt
            W(:, iMap(k, :), t) = W0(:, :, t);
        end
        [~, Wh0] = max(sq(sum(W.^2, 1)), [], 1);
        
        % pairs of templates that live on the same channels are potential "matches"
        iMatch = sq(min(abs(single(iC) - reshape(iMap(k, Wh0), 1, 1, [])), [], 1))<.1;
        
        % compute dissimilarities for iMatch = 1
        [iclust, ds] = mexDistances2(Params, Ws, W, iMatch, iC-1, Whs-1, mus, sq(mu0));
        
        % ds are squared Euclidian distances
        iclustall(:,:,k) = reshape(iclust, Nfilt, []) + 1;
        ds = reshape(ds, Nfilt, []); % this should just be an Nfilt-long vector
        ds = max(0, ds);
        cc0(k,:) = mean(sqrt(ds) .* ns, 1)./mean(ns,1); % weigh the distances according to number of spikes in cluster
        
    end
    
    [dmin, imin] = min(cc0, [], 1);
    medshift = mode(imin);
    irange = find(imin==medshift);
    [~, isort] = sort(dmin(irange), 'ascend');
    irange = irange(isort(1:50));
    
    W0  = gpuArray.zeros(nPCs , Nchan, Nfilt, 'single');
    nn = 1e-4 * ones(Nfilt,1);
    Ns = zeros(Nfilt,1);
    for j = 1:length(irange)
        ibatch = irange(j);
        icl = iclustall(:, ibatch, medshift);
        for t = 1:Nfilt
            if icl(t)>0
                W0(:,iC(:, Whs(t,ibatch)), icl(t)) =  W0(:,iC(:, Whs(t,ibatch)), icl(t)) + ...
                    mus(t, ibatch) * Ws(:,:,t,ibatch) ;
                nn(icl(t)) = nn(icl(t)) + 1;
                Ns(icl(t)) = Ns(icl(t)) + gather(ns(t, ibatch));
            end
        end
    end
    for t = 1:Nfilt
        W0(:,:,t) = W0(:,:,t) / nn(t);
    end
    mu0 = 1e-3 + sum(sum(W0.^2,1),2).^.5;
    W0 = W0 ./ mu0;
    Ns = Ns./nn;
%     mu0 = sq(mu0);
end

W0 = W0 .* mu0;

