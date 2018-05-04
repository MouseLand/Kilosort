function rez = fwbwClustering(rez)

nPCs    = rez.ops.nPCs;
Nfilt   = rez.ops.Nfilt;
nfullpasses = rez.ops.nfullpasses;

% extract PC projections here
tic
wPCA    = extractPCfromSnippets(rez, nPCs);
fprintf('Obtained 7 PC waveforms in %2.2f seconds \n', toc)

tic
[uprojDAT, call, muall, tsall]  = extractPCfeatures0(rez, wPCA);
fprintf('Extracted %d spikes with %d features in %2.2f seconds \n', ...
    size(uprojDAT,2), size(uprojDAT,1), toc)

tic
nch     = 8;
nCHmax  = max(call) + nch;

ioff = nPCs * gpuArray(int32(call - 8));

% split into batches for GPU processing
nS       = size(uprojDAT,2); % number of spikes
nSamples             = 1e4; % number of spikes per batch
[iBatch, nBatches]   = makeBatches(nS, nSamples); 
for k = 1:nBatches
   [~, isort] = sort(call(iBatch{k}));
   iBatch{k} = iBatch{k}(isort);
end

% iBatch = iBatch(randperm(numel(iBatch)));

[W, Wheights]       = initializeW(Nfilt, nCHmax, nPCs);
iorig   = 1:Nfilt;

pm      = exp(-1/200); % momentum term
Nnearest = 32;
Params  = [nSamples size(uprojDAT,1) Nfilt pm size(W,1) 0 Nnearest];

irounds = [1:nBatches nBatches:-1:1]; 
Boffset = 0; %ceil(nBatches/2);
niter   = nfullpasses * 2*nBatches + Boffset; 
ops.lam = [5 30];
lami    = exp(linspace(log(ops.lam(1)), log(ops.lam(2)), niter));

flag_resort      = 0;
flag_initialized = 0;

mu      = gpuArray.ones(Nfilt, 1, 'single');
dWU     = gpuArray.zeros(size(W), 'single');
M       = NaN * gpuArray.ones(nS,1, 'single');
iList   = gpuArray(int32(ones(Nnearest, Nfilt)));
cfall   = zeros(nS, Nnearest, 'single');
nspfilt = zeros(1, Nfilt, 'single');
nspmax  = zeros(1, Nfilt, 'single');
Cost    = zeros(niter,1);
icl     = zeros(nS, 1);

tauD = 2;

for i = 1:niter
    k = irounds(rem(i-1 + Boffset, 2*nBatches)+1);
    
    uproj = gpuArray(uprojDAT(:, iBatch{k}));
    Params(1) = size(uproj,2);
    
    %  boolean variable: should we compute spike x filter
    iW = abs(call(iBatch{k})' - Wheights) < 10;
    ioffsub = ioff(iBatch{k});
    
    % get iclust and update W
    [dWU, iclust, cmax, cf, nsp] = mexClustering(Params, uproj, W, ioffsub, ...
        iW, dWU, mu, iList-1);  
    
%     nspfilt(iorig) = exp(-1/tauD) * nspfilt(iorig) + (1 - exp(-1/tauD)) * single(nsp');
%     nspmax = max(nspfilt, nspmax);
    
%     coefs(iBatch{k}, :) = cmax;
    
    M(iBatch{k}) = max(cmax, [], 2);
    
    icl(iBatch{k}) = iorig(iclust+1); 
    
    % update W
    if i==100
       flag_resort = 1; 
       flag_initialized = 1;
    end

    if flag_initialized
        mu = sum(dWU.^2,1).^.5;
        W = dWU./(1e-5 + mu);
        Params(6) = lami(i);
        if ~flag_resort
            cfall(iBatch{k}, :) = cf; 
        end
    end

    if flag_resort
        W = reshape(W, nPCs, nCHmax, Nfilt);
        nW = sq(sum(W(1, :, :).^2,1));
        W = reshape(W, nCHmax * nPCs, Nfilt);
                
        [~, Wheights] = max(nW,[], 1);
        [Wheights, isort] = sort(Wheights);
        iorig = iorig(isort); 
        W = W(:, isort);
        dWU = dWU(:, isort);
    end
    
    
    if i==niter-nBatches
        flag_resort = 0;
        cc = W' * W;
        [~, isort] = sort(cc, 1, 'descend');
        iList = int32(gpuArray(isort(1:Nnearest, :)));
    end
    
%     if rem(i,100)==1
%        p = p+1;
%        Cost(p) = gather(nanmean(M.^2));
%        plot(Cost(1:p))
%        drawnow
%     end
end


iresort(iorig) = 1:Nfilt;

rez.cfall   = cfall - cfall(:,1);
rez.st      = tsall;
rez.clu     = iresort(icl);
rez.wPCA    = wPCA;
rez.W       = W;
rez.iList   = iList;
rez.call    = call;

fprintf('Optimization complete in %2.2f seconds \n', toc)

%%
% S = sparse(ceil(((tsall -tsall(1))+1)/3e4), icl, ones(1, numel(icl)));
% S(1, Nfilt) = 0;
% 
% S = gpuArray(single(full(S)));
% 
% % Shigh = S - mean(S,1);
% % Shigh = S - my_conv2(S,500,1);
% 
% Slow = my_conv2(S,500,1);
% 
% rat = min(Slow, [], 1) ./max(Slow, [],1);
% 
% S0 = S(:, rat>.5);
% 
% % [U Sv V] = svdecon(zscore(Shigh, 1, 1));
% [U Sv V] = svdecon(S0 - mean(S0, 1));
% 
% clf
% plot(U(:,1:4))
% %%
% clear iresort
% iresort(iorig) = 1:Nfilt;
% imagesc(S(:, iresort), [0 20])
% 
