function rez = clusterSingleBatches(rez)
% outputs an ordering of the batches according to drift
% for each batch, it extracts spikes as threshold crossings and clusters them with kmeans
% the resulting cluster means are then compared for all pairs of batches, and a dissimilarity score is assigned to each pair
% the matrix of similarity scores is then re-ordered so that low dissimilaity is along the diagonal


rng('default'); rng(1);

ops = rez.ops;

if getOr(ops, 'reorder', 0)==0
    rez.iorig = 1:rez.temp.Nbatch; % if reordering is turned off, return consecutive order
    return;
end

nPCs    = getOr(rez.ops, 'nPCs', 3);
Nfilt = ceil(rez.ops.Nchan/2);
tic
wPCA    = extractPCfromSnippets(rez, nPCs); % extract PCA waveforms pooled over channels
fprintf('Obtained 7 PC waveforms in %2.2f seconds \n', toc) % 7 is the default, and I don't think it needs to be able to change

Nchan = rez.ops.Nchan;
niter = 10; % iterations for k-means. we won't run it to convergence to save time

nBatches      = rez.temp.Nbatch;
NchanNear = min(Nchan, 2*8+1);

% initialize big arrays on the GPU to hold the results from each batch
Ws = gpuArray.zeros(nPCs , NchanNear, Nfilt, nBatches, 'single'); % this holds the unit norm templates
mus = gpuArray.zeros(Nfilt, nBatches, 'single'); % this holds the scalings
ns = gpuArray.zeros(Nfilt, nBatches, 'single'); % this holds the number of spikes for that cluster
Whs = gpuArray.ones(Nfilt, nBatches, 'int32'); % this holds the center channel for each template

i0 = 0;

NrankPC = 3; % I am not sure if this gets used, but it goes into the function

iC = getClosestChannels(rez, ops.sigmaMask, NchanNear); % return an array of closest channels for each channel

tic
for ibatch = 1:nBatches
    [uproj, call] = extractPCbatch2(rez, wPCA, min(nBatches-1, ibatch), iC); % extract spikes using PCA waveforms
    % call contains the center channels for each spike

    if sum(isnan(uproj(:)))>0 %sum(mus(:,ibatch)<.1)>30
        break; % I am not sure what case this safeguards against....
    end

    if size(uproj,2)>Nfilt
       % if a batch has at least as many spikes as templates we request, then cluster it
        [W, mu, Wheights, irand] = initializeWdata2(call, uproj, Nchan, nPCs, Nfilt, iC); % this initialize the k-means

        % Params is a whole bunch of parameters sent to the C++ scripts inside a float64 vector
        Params  = [size(uproj,2) NrankPC Nfilt 0 size(W,1) 0 NchanNear Nchan];

        for i = 1:niter
            Wheights = reshape(Wheights, 1,1,[]); % this gets reshaped for broadcasting purposes
            % we only compute distances to clusters on the same channels
            iMatch = sq(min(abs(single(iC) - Wheights), [], 1))<.1; % this tells us which spikes and which clusters might match

            % get iclust and update W
            [dWU, iclust, dx, nsp, dV] = mexClustering2(Params, uproj, W, mu, ...
                call-1, iMatch, iC-1); % CUDA script to efficiently compute distances for pairs in which iMatch is 1

            dWU = dWU./(1e-5 + single(nsp')); % divide the cumulative waveform by the number of spikes

            mu = sum(dWU.^2,1).^.5; % norm of cluster template
            W = dWU./(1e-5 + mu); % unit normalize templates

            W = reshape(W, nPCs, Nchan, Nfilt);
            nW = sq(W(1, :, :).^2); % compute best channel from the square of the first PC feature
            W = reshape(W, Nchan * nPCs, Nfilt);

            [~, Wheights] = max(nW,[], 1); % the new best channel of each cluster template
        end

        % carefully keep track of cluster templates in dense format
        W = reshape(W, nPCs, Nchan, Nfilt);
        W0 = gpuArray.zeros(nPCs, NchanNear, Nfilt, 'single');
        for t = 1:Nfilt
            W0(:, :, t) = W(:, iC(:, Wheights(t)), t);
        end
        W0 = W0 ./ (1e-5 + sum(sum(W0.^2,1),2).^.5); % I don't really know why this needs another normalization
    end

    if exist('W0', 'var')
        % if a batch doesn't have enough spikes, it gets the cluster templates of the previous batch
        Ws(:, :, :, ibatch)   = W0;
        mus(:, ibatch)     = mu;
        ns(:, ibatch)      = nsp;
        Whs(:, ibatch)     = int32(Wheights);
    else
      % if the first batch doesn't have enough spikes, then it is skipped completely
        warning('data batch #%d only had %d spikes \n', ibatch, size(uproj,2))
    end
    i0 = i0 + Nfilt;

    if rem(ibatch, 500)==1
        fprintf('time %2.2f, pre clustered %d / %d batches \n', toc, ibatch, nBatches)
    end
end


tic
% anothr one of these Params variables transporting parameters to the C++ code
Params  = [1 NrankPC Nfilt 0 size(W,1) 0 NchanNear Nchan];
Params(1) = size(Ws,3) * size(Ws,4); % the total number of templates is the number of templates per batch times the number of batches

% initialize dissimilarity matrix
ccb = gpuArray.zeros(nBatches, 'single');

for ibatch = 1:nBatches
    % for every batch, compute in parallel its dissimilarity to ALL other batches
    Wh0 = single(Whs(:, ibatch)); % this one is the primary batch
    W0  = Ws(:, :, ibatch);
    mu = mus(:, ibatch);

    % embed the templates from the primary batch back into a full, sparse representation
    W = gpuArray.zeros(nPCs , Nchan, Nfilt, 'single');
    for t = 1:Nfilt
        W(:, iC(:, Wh0(t)), t) = Ws(:, :, t, ibatch);
    end

    % pairs of templates that live on the same channels are potential "matches"
    iMatch = sq(min(abs(single(iC) - reshape(Wh0, 1, 1, [])), [], 1))<.1;


    % compute dissimilarities for iMatch = 1
    [iclust, ds] = mexDistances2(Params, Ws, W, iMatch, iC-1, Whs-1, mus, mu);

    % ds are squared Euclidian distances
    ds = reshape(ds, Nfilt, []); % this should just be an Nfilt-long vector
    ds = max(0, ds);
    ccb(ibatch,:) = mean(sqrt(ds) .* ns, 1)./mean(ns,1); % weigh the distances according to number of spikes in cluster

    if rem(ibatch, 500)==1
        fprintf('time %2.2f, compared %d / %d batches \n', toc, ibatch, nBatches)
    end
end

% some normalization steps are needed: zscoring, and symmetrizing ccb
ccb0 = zscore(ccb, 1, 1);
ccb0 = ccb0 + ccb0';

rez.ccb = gather(ccb0);

% sort by manifold embedding algorithm
% iorig is the sorting of the batches
% ccbsort is the resorted matrix (useful for diagnosing drift)
[ccbsort, iorig] = sortBatches2(ccb0);

% some mandatory diagnostic plots to understand drift in this dataset
figure;
subplot(1,2,1)
imagesc(ccb0, [-5 5]); drawnow
xlabel('batches')
ylabel('batches')
title('batch to batch distance')

subplot(1,2,2)
imagesc(ccbsort, [-5 5]); drawnow
xlabel('sorted batches')
ylabel('sorted batches')
title('AFTER sorting')

rez.iorig = gather(iorig);
rez.ccbsort = gather(ccbsort);

fprintf('time %2.2f, Re-ordered %d batches. \n', toc, nBatches)
%%
