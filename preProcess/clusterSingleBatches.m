% function rez = clusterSingleBatches(rez)

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
niter = 10;    

tic
for ibatch = 1:niter    
    [uproj, call, muall, tsall] = extractPCbatch(rez, wPCA, ibatch);
    

    nCHmax  = max(call) + nch;
    
    ioff = nPCs * gpuArray(int32(call - nch - 1));
    
    % [W, Wheights]       = initializeW(Nfilt, nCHmax, nPCs);
    [W, mu, Wheights] = initializeWdata(ioff, uproj, nCHmax, nPCs, Nfilt);
    
    Params  = [1 size(uproj,1) Nfilt pm size(W,1) 0 Nnearest];
    Params(1) = size(uproj,2);
    
    for i = 1:niter
        dWU     = gpuArray.zeros(size(W), 'single');
        
        
        %  boolean variable: should we compute spike x filter
        iW = abs(call - Wheights) < 10;
        
        % get iclust and update W
        [dWU, iclust, cmax, nsp] = mexClustering(Params, uproj, W, ioff, ...
            iW, dWU, mu);
        
        dWU = dWU./single(nsp');
        
        mu = sum(dWU.^2,1).^.5;
        W = dWU./(1e-5 + mu);
    end
end
toc
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
