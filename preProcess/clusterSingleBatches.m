function rez = clusterSingleBatches(rez)

nPCs    = getOr(rez.ops, 'nPCs', 3);
% Nfilt   = rez.ops.Nfilt;
Nfilt = ceil(rez.ops.Nchan/2);


% extract PC projections here
tic
wPCA    = extractPCfromSnippets(rez, nPCs);
fprintf('Obtained 7 PC waveforms in %2.2f seconds \n', toc)

nch     = 8;
pm      = 0; % momentum term
Nnearest = 32;

Nchan = rez.ops.Nchan;
niter = 10;

nBatches      = rez.temp.Nbatch;

Ws = gpuArray.zeros(nPCs * (2*nch+1), Nfilt, nBatches, 'single');
mus = gpuArray.zeros(Nfilt, nBatches, 'single');
ns = gpuArray.zeros(Nfilt, nBatches, 'single');
ib = gpuArray.zeros(Nfilt, nBatches, 'int32');
ioffWs = gpuArray.zeros(Nfilt, nBatches, 'int32');
Whs = gpuArray.zeros(Nfilt, nBatches, 'single');

i0 = 0;

tic
for ibatch = 1:nBatches        
    [uproj, call, muall, tsall] = extractPCbatch(rez, wPCA, min(nBatches-1, ibatch));  
    
    ioff = nPCs * gpuArray(int32(call - nch - 1));
    
    [W, mu, Wheights] = initializeWdata(ioff, uproj, Nchan, nPCs, Nfilt);
    
    Params  = [1 size(uproj,1) Nfilt pm size(W,1) 0 Nnearest];
    Params(1) = size(uproj,2);
    
    for i = 1:niter
        dWU     = gpuArray.zeros(size(W), 'single');        
        
        %  boolean variable: should we compute spike x filter
        iW = abs(call - Wheights) < 10;
        
        % get iclust and update W
        [dWU, iclust, dx, nsp, dV] = mexClustering(Params, uproj, W, ioff, ...
            iW, dWU, mu);
        
        dWU = dWU./(1e-5 + single(nsp'));
        
%         [dWU, nnew] = triageKmeans(dWU, nsp, uproj,ioff, dx./nu);
        
        mu = sum(dWU.^2,1).^.5;
        W = dWU./(1e-5 + mu);
        
        W = reshape(W, nPCs, Nchan, Nfilt);
        nW = sq(sum(W(1, :, :).^2,1));
        W = reshape(W, Nchan * nPCs, Nfilt);
        
        [~, Wheights] = max(nW,[], 1);
        [Wheights, isort] = sort(Wheights);
        
        W   = W(:, isort);
        mu  = mu(isort);
    end

    Wheights = min(max(Wheights, nch+1), Nchan-nch);
    ioffW = nPCs * double(Wheights - nch - 1);
    
    W0 = reshape(W(ioffW + [1:size(uproj,1)]' + ([1:Nfilt]-1)*size(W,1)),...
        size(uproj,1), Nfilt);
    
    W0 = W0 ./ (1e-5 + sum(W0.^2,1).^.5);
    
    Ws(:, :, ibatch)   = W0;    
    mus(:, ibatch)     = mu;
    ns(:, ibatch)      = nsp;    
    ib(:, ibatch)      = ibatch;
    ioffWs(:, ibatch)  = ioffW;
    Whs(:, ibatch)     = Wheights;
    
    i0 = i0 + Nfilt;
    
    if rem(ibatch, 500)==1
        fprintf('time %2.2f, pre clustered %d / %d batches \n', toc, ibatch, nBatches)
    end
end

ns = reshape(ns, Nfilt, []);

Params  = [1 size(uproj,1) Nfilt pm size(W,1) 0 Nnearest];
Params(1) = size(Ws,2) * size(Ws,3);

ccb = gpuArray.zeros(nBatches, 'single');

[ncoefs, Nfilt, nBatches] = size(Ws);
for ibatch = 1:nBatches 
    Wh0 = Whs(:, ibatch);    
    W0  = Ws(:, :, ibatch);    
    
    W = gpuArray.zeros(nPCs * Nchan, Nfilt, 'single');
    ioffW = double(ioffWs(:, ibatch))';
    irange = [1:ncoefs]' + ioffW;
    
    W(irange + [0:Nfilt-1]*nPCs * Nchan) = W0;    
    
    iW = abs(Whs(:) - Wh0') < 5;    
    
    mu = mus(:, ibatch);
    
    [iclust, ds] = mexDistances(Params, Ws, W, ioffWs(:), iW, mus, mu);
    
    ds = reshape(ds, Nfilt, []);
    ds = max(0, ds);
    
    ccb(ibatch,:) = mean(sqrt(ds) .* ns, 1)./mean(ns,1);
end


ccb0 = zscore(ccb, 1, 1);
ccb0 = ccb0 + ccb0';
rez.ccb = gather(ccb0);

[u, s, v] = svdecon(ccb0);
[~, isort] = sort(u(:,1));
iorig = isort;    
ccb0 = ccb0(isort, isort);

rez.iorig = iorig;

fprintf('time %2.2f, Re-ordered %d batches. \n', toc, nBatches)
%% 
