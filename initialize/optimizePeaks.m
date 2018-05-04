% addpath('C:\CODE\GitHub\KiloSort\preDetect')
function WUinit=optimizePeaks(ops,uproj)

nt0             = ops.nt0;

nProj           = size(uproj,2);
nSpikesPerBatch = 4000;
inds            = 1:nSpikesPerBatch * floor(size(uproj,1)/nSpikesPerBatch);
inds            = reshape(inds, nSpikesPerBatch, []);
% Nbatch = size(inds,2);
iperm           = randperm(size(inds,2));
miniorder       = repmat(iperm, 1, ops.nfullpasses);
%     miniorder = repmat([1:Nbatch Nbatch:-1:1], 1, ops.nfullpasses/2);

if ~exist('spikes_merged')
    uBase = zeros(1e4, nProj);
    nS = zeros(1e4, 1);
    ncurr = 1;
    
    for ibatch = 1:size(inds,2)
        % merge in with existing templates
        uS = uproj(inds(:,ibatch), :);
        [nSnew, iNonMatch] = merge_spikes0(uBase(1:ncurr,:), nS(1:ncurr), uS, ops.crit);
        nS(1:ncurr) = nSnew;
        %
        % reduce non-matches
        [uNew, nSadd] = reduce_clusters0(uS(iNonMatch,:), ops.crit);
        
        % add new spikes to list
        uBase(ncurr + [1:size(uNew,1)], :) = uNew;
        nS(ncurr + [1:size(uNew,1)]) = nSadd;
        
        ncurr = ncurr + size(uNew,1);
        
        if ncurr>1e4
            break;
        end
    end
    %
    nS = nS(1:ncurr);
    uBase = uBase(1:ncurr, :);
    spikes_merged = 1;
end
[~, itsort] = sort(nS, 'descend');

%% initialize U
Nfilt = ops.Nfilt;
lam = ops.lam(1) * ones(Nfilt, 1, 'single');

ind_filt = itsort(rem([1:Nfilt]-1, numel(itsort)) + 1);
if ops.GPU
    U = gpuArray(uBase(ind_filt, :))';
else
    U = uBase(ind_filt, :)';
end
U = U + .001 * randn(size(U));
mu = sum(U.^2,1)'.^.5;
U = normc(U);
%

for i = 1:10
    
    idT = zeros(size(inds));
    dWU = zeros(Nfilt, nProj, 'single');
    if ops.GPU
        nToT = gpuArray.zeros(Nfilt, 1, 'single');
        Cost = gpuArray(single(0));
    else
        nToT = zeros(Nfilt, 1, 'single');
        Cost = single(0);
    end
    
    for ibatch = 1:size(inds,2)
        % find clusters
        if ops.GPU
            clips = reshape(gpuArray(uproj(inds(:,ibatch), :)), nSpikesPerBatch, nProj);
        else
            clips = reshape(uproj(inds(:,ibatch), :), nSpikesPerBatch, nProj);
        end
        ci = clips * U;
        
        ci = bsxfun(@plus, ci, (mu .* lam)');
        cf = bsxfun(@rdivide, ci.^2, 1 + lam');
        cf = bsxfun(@minus, cf, (mu.^2.*lam)');
        
        [max_cf, id] = max(cf, [], 2);
        
        id = gather_try(id);
        %        x = ci([1:nSpikesPerBatch] + nSpikesPerBatch * (id-1)')' - mu(id) .* lam(id);
        idT(:,ibatch) = id;
        
        if ops.GPU
            L = gpuArray.zeros(Nfilt, nSpikesPerBatch, 'single');
        else
            L = zeros(Nfilt, nSpikesPerBatch, 'single');
        end
        L(id' + [0:Nfilt:(Nfilt*nSpikesPerBatch-1)]) = 1;
        dWU = dWU + L * clips;
        
        nToT = nToT + sum(L, 2);
        Cost = Cost + mean(max_cf);
    end
    dWU  = bsxfun(@rdivide, dWU, nToT);
    
    U = dWU';
    mu = sum(U.^2,1)'.^.5;
    U = normc(U);
    Cost = Cost/size(inds,2);
    
%     disp(Cost)
    
%     plot(sort(log(1+nToT)))
%     drawnow
end
%%
Nchan = ops.Nchan;
Nfilt = ops.Nfilt;
Nrank = ops.Nrank;
wPCA = ops.wPCA(:,1:3);
Urec = reshape(U, Nchan, size(wPCA,2), Nfilt);

Urec= permute(Urec, [2 1 3]);
Wrec = reshape(wPCA * Urec(:,:), nt0, Nchan, Nfilt);

Wrec = gather_try(Wrec);

W = zeros(nt0, Nfilt, Nrank, 'single');
U = zeros(Nchan, Nfilt, Nrank, 'single');

Wrec(isnan(Wrec(:))) = 0;
for j = 1:Nfilt
    [w sv u] = svd(Wrec(:,:,j));
    w = w * sv;
    
    Sv = diag(sv);
    W(:,j,:) = w(:, 1:Nrank)/sum(Sv(1:ops.Nrank).^2).^.5;
    U(:,j,:) = u(:, 1:Nrank);
end

Uinit = U;
Winit = W;
mu = gather_try(single(mu));
muinit = mu;

WUinit = zeros(nt0, Nchan, Nfilt);
for j = 1:Nfilt
    WUinit(:,:,j) = muinit(j)  * Wrec(:,:,j);
end
WUinit = single(WUinit);
%%


