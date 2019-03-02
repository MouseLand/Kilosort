function rez = clusterSingleBatches(rez)
rng('default'); rng(1);

ops = rez.ops;

nPCs    = getOr(rez.ops, 'nPCs', 3);
% Nfilt   = rez.ops.Nfilt;
Nfilt = ceil(rez.ops.Nchan/2);

% extract PC projections here
tic

wPCA    = extractPCfromSnippets(rez, nPCs);
fprintf('Obtained 7 PC waveforms in %2.2f seconds \n', toc)

% nch     = 8;
% pm      = 0; % momentum term
% Nnearest = 32;

Nchan = rez.ops.Nchan;
niter = 10;

nBatches      = rez.temp.Nbatch;

NchanNear = min(Nchan, 2*8+1);

Ws = gpuArray.zeros(nPCs , NchanNear, Nfilt, nBatches, 'single');
mus = gpuArray.zeros(Nfilt, nBatches, 'single');
ns = gpuArray.zeros(Nfilt, nBatches, 'single');
Whs = gpuArray.zeros(Nfilt, nBatches, 'int32');

i0 = 0;

NrankPC = 3;

iC = getClosestChannels(rez, ops.sigmaMask, NchanNear);

% initialize W0, mu, nsp, just in case the first batch is empty
Whs = gpuArray.ones(Nfilt, nBatches, 'int32');

tic
for ibatch = 1:nBatches            
    [uproj, call] = extractPCbatch2(rez, wPCA, min(nBatches-1, ibatch), iC);  
    
    if sum(isnan(uproj(:)))>0 %sum(mus(:,ibatch)<.1)>30
        break;
    end
    
    if size(uproj,2)>Nfilt
        [W, mu, Wheights, irand] = initializeWdata2(call, uproj, Nchan, nPCs, Nfilt, iC);
        
        Params  = [size(uproj,2) NrankPC Nfilt 0 size(W,1) 0 NchanNear Nchan];        
        
        Wheights = reshape(Wheights, 1,1,[]);
        
        for i = 1:niter
            Wheights = reshape(Wheights, 1,1,[]);
            iMatch = sq(min(abs(single(iC) - Wheights), [], 1))<.1;

            % get iclust and update W
            [dWU, iclust, dx, nsp, dV] = mexClustering2(Params, uproj, W, mu, ...
                call-1, iMatch, iC-1);
            
            dWU = dWU./(1e-5 + single(nsp'));
            
            mu = sum(dWU.^2,1).^.5;
            W = dWU./(1e-5 + mu);
            
            W = reshape(W, nPCs, Nchan, Nfilt);
            nW = sq(sum(W(1, :, :).^2,1));
            W = reshape(W, Nchan * nPCs, Nfilt);
            
            [~, Wheights] = max(nW,[], 1);
        end
                
        W = reshape(W, nPCs, Nchan, Nfilt);
        W0 = gpuArray.zeros(nPCs, NchanNear, Nfilt, 'single');
        for t = 1:Nfilt
            W0(:, :, t) = W(:, iC(:, Wheights(t)), t);
        end        
        W0 = W0 ./ (1e-5 + sum(sum(W0.^2,1),2).^.5);
    end

    if exist('W0', 'var')
        Ws(:, :, :, ibatch)   = W0;
        mus(:, ibatch)     = mu;
        ns(:, ibatch)      = nsp;
        Whs(:, ibatch)     = int32(Wheights);
    else
        warning('data batch #%d only had %d spikes \n', ibatch, size(uproj,2))
    end    
    i0 = i0 + Nfilt;    
    
    if rem(ibatch, 500)==1
        fprintf('time %2.2f, pre clustered %d / %d batches \n', toc, ibatch, nBatches)
    end
end


tic
ns = reshape(ns, Nfilt, []);

Params  = [1 NrankPC Nfilt 0 size(W,1) 0 NchanNear Nchan];
Params(1) = size(Ws,3) * size(Ws,4);

ccb = gpuArray.zeros(nBatches, 'single');
d1 = gpuArray.zeros(nBatches, 'single');

% [ncoefs, Nfilt, nBatches] = size(Ws);
for ibatch = 1:nBatches 
    Wh0 = single(Whs(:, ibatch));    
    W0  = Ws(:, :, ibatch);    
    mu = mus(:, ibatch);
    
    W = gpuArray.zeros(nPCs , Nchan, Nfilt, 'single');
    for t = 1:Nfilt
        W(:, iC(:, Wh0(t)), t) = Ws(:, :, t, ibatch);
    end
    % this gets replaced with imatch
    iMatch = sq(min(abs(single(iC) - reshape(Wh0, 1, 1, [])), [], 1))<.1;
    
    imin = ones(1, nBatches);
    cct = Inf * gpuArray.ones(1, nBatches, 'single');    
    
    %----------------------------------%    
    [iclust, ds] = mexDistances2(Params, Ws, W, iMatch, iC-1, Whs-1, mus, mu);
    
    ds = reshape(ds, Nfilt, []);
    ds = max(0, ds);    
    cct0 = mean(sqrt(ds) .* ns, 1)./mean(ns,1);    
    ix = cct0 < cct;
    cct(ix) = cct0(ix);
    imin(ix) = t;
    %----------------------------------%
    
    ccb(ibatch,:) = cct;
    d1(ibatch,:) = imin;
    
    if rem(ibatch, 500)==1
        fprintf('time %2.2f, compared %d / %d batches \n', toc, ibatch, nBatches)
    end
end

ccb0 = zscore(ccb, 1, 1);
ccb0 = ccb0 + ccb0';

rez.ccb = gather(ccb0);

% sort by new manifold algorithm
[ccb1, iorig] = sortBatches2(ccb0);

figure;
subplot(1,2,1)
imagesc(ccb0, [-5 5]); drawnow
xlabel('batches')
ylabel('batches')
title('batch to batch distance')

subplot(1,2,2)
imagesc(ccb1, [-5 5]); drawnow
xlabel('sorted batches')
ylabel('sorted batches')
title('AFTER sorting')

rez.iorig = gather(iorig);
rez.ccbsort = gather(ccb1);

fprintf('time %2.2f, Re-ordered %d batches. \n', toc, nBatches)
%% 

