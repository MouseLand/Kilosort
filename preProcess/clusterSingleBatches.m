function rez = clusterSingleBatches(rez)

nPCs    = rez.ops.nPCs;
% Nfilt   = rez.ops.Nfilt;
Nfilt = 128;


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


flag_resort = 1;

for ibatch = 1:nBatches    
    [uproj, call, muall, tsall] = extractPCbatch(rez, wPCA, ibatch);  

    
    ioff = nPCs * gpuArray(int32(call - nch - 1));
    
    % [W, Wheights]       = initializeW(Nfilt, nCHmax, nPCs);
    [W, mu, Wheights] = initializeWdata(ioff, uproj, Nchan, nPCs, Nfilt);
    
    Params  = [1 size(uproj,1) Nfilt pm size(W,1) 0 Nnearest];
    Params(1) = size(uproj,2);
    
    for i = 1:niter
        dWU     = gpuArray.zeros(size(W), 'single');
        
        
        %  boolean variable: should we compute spike x filter
        iW = abs(call - Wheights) < 10;
        
        % get iclust and update W
        [dWU, iclust, cmax, nsp] = mexClustering(Params, uproj, W, ioff, ...
            iW, dWU, mu);
        
        dWU = dWU./(1e-5 + single(nsp'));
        
        mu = sum(dWU.^2,1).^.5;
        W = dWU./(1e-5 + mu);
        
         if flag_resort
            W = reshape(W, nPCs, Nchan, Nfilt);
            nW = sq(sum(W(1, :, :).^2,1));
            W = reshape(W, Nchan * nPCs, Nfilt);
            
            [~, Wheights] = max(nW,[], 1);
            [Wheights, isort] = sort(Wheights);            
            
            W   = W(:, isort);
            dWU = dWU(:, isort);
            mu  = mu(isort);
         end
    end

    Wheights = min(max(Wheights, nch+1), Nchan-nch);
    ioffW = nPCs * double(Wheights - nch - 1);
    
    W0 = reshape(W(ioffW + [1:size(uproj,1)]' + ([1:Nfilt]-1)*size(W,1)),...
        size(uproj,1), Nfilt);
    
    Ws(:, :, ibatch)   = W0;    
    mus(:, ibatch)     = mu;
    ns(:, ibatch)      = nsp;    
    ib(:, ibatch)      = ibatch;
    ioffWs(:, ibatch)  = ioffW;
    Whs(:, ibatch)     = Wheights;
    
    i0 = i0 + Nfilt;
end

%%
ns = reshape(ns, Nfilt, []);

Params  = [1 size(uproj,1) Nfilt pm size(W,1) 0 Nnearest];
Params(1) = size(Ws,2) * size(Ws,3);

ccb = gpuArray.zeros(nBatches, 'single');


for ibatch = 1:nBatches    %:nBatches            
    Wh0 = Whs(:, ibatch);    
    W0  = Ws(:, :, ibatch);
    
    nfilters = size(W0,2);
    
    W = gpuArray.zeros(nPCs * Nchan, nfilters, 'single');
    W(double(ioffWs(:, ibatch))' + [1:size(uproj,1)]' + ...
        ([1:nfilters]-1)*size(W,1)) = W0;    
    
    iW = abs(Whs(:) - Wh0') < 5;
    
    mu = mus(:, ibatch);
    
    [iclust, ds] = mexDistances(Params, Ws, W, ioffWs(:), iW, mus, mu);
    
    ds = reshape(ds, Nfilt, []);
    ccb(ibatch,:) = mean(ds .* ns, 1)./mean(ns,1);
end

ccb0 = zscore(ccb, 1, 2);
ccb0 = ccb0 + ccb0';

iorig = 1:nBatches;
for t = 1:50
    ccs = my_conv2(ccb0, 100, 1);
    ccs = zscore(ccs, 1, 2)/nBatches;
    ch = ccb0 * ccs';
    
    [~, imax]  = max(ch, [], 2);
    [~, isort] = sort(imax);
    
    iorig = iorig(isort);    
    
    ccb0 = ccb0(isort, isort);
end

rez.iorig = iorig;
rez.ccb = gather(ccb);
%% 
