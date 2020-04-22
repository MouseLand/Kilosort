function imin = find_integer_shifts(Params, Whs,Ws,mus, ns, iC, Nchan, Nfilt)

nBatches = size(Whs, 2);
nPCs = size(Ws, 1);
NchanNear = size(Ws, 2);

ibatch0 = min(300, floor(nBatches/2));

iChan = 1:Nchan;
iUp = mod(iChan + 2-1, Nchan)+1;
iDown = mod(iChan - 2-1, Nchan)+1;
iMap = [iUp(iUp); iUp; iChan; iDown; iDown(iDown)];

% for every batch, compute in parallel its dissimilarity to ALL other batches
Wh0 = single(Whs(:, ibatch0)); % this one is the primary batch
W0  = Ws(:, :, :, ibatch0);
mu0  = mus(:, ibatch0);

for iter = 1:10
    cc0 = gpuArray.zeros(5, nBatches);
    for k = 1:5
        % embed the templates from the primary batch back into a full, sparse representation
        W = gpuArray.zeros(nPCs , Nchan, Nfilt, 'single');
        for t = 1:Nfilt
            W(:, iMap(k, iC(:, Wh0(t))), t) = W0(:, :, t);
        end
        
        % pairs of templates that live on the same channels are potential "matches"
        iMatch = sq(min(abs(single(iC) - reshape(iMap(k, Wh0), 1, 1, [])), [], 1))<.1;
        
        % compute dissimilarities for iMatch = 1
        [iclust, ds] = mexDistances2(Params, Ws, W, iMatch, iC-1, Whs-1, mus, mu0);
        
        % ds are squared Euclidian distances
        iclust = reshape(iclust, Nfilt, []);
        ds = reshape(ds, Nfilt, []); % this should just be an Nfilt-long vector
        ds = max(0, ds);
        cc0(k,:) = mean(sqrt(ds) .* ns, 1)./mean(ns,1); % weigh the distances according to number of spikes in cluster
        
    end
    
    iclust = iclust + 1;
    
    [~, imin] = min(cc0, [], 1);
    irange = find(imin==median(imin));
    W0 = gpuArray.zeros(nPCs , NchanNear, Nfilt, 'single');
    mu0 = gpuArray.zeros(Nfilt, 1, 'single');
    
    nn = zeros(Nfilt,1);
    for j = 1:length(irange)
        ibatch = irange(j);
        icl = iclust(:, ibatch);
        W0(:,:,icl>0) =  W0(:,:,icl>0) + Ws(:,:,icl(icl>0),ibatch) ;
        nn(icl>0) = nn(icl>0) + 1;
        
        mu0(icl>0) = mu0(icl>0) + mus(icl(icl>0), ibatch);
    end
    mu0 = mu0 ./ nn;
    for t = 1:Nfilt
        W0(:,:,t) = W0(:,:,t) / nn(t);
    end
    nm = sum(sum(W0.^2,1),2).^.5;
    W0 = W0 ./ nm;
    mu0 = mu0 .* nm(:);
end


