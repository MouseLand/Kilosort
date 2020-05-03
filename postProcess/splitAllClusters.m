function [rez, X] = splitAllClusters(rez, flag)
% i call this algorithm "bimodal pursuit"
% split clusters if they have bimodal projections
% the strategy is to maximize a bimodality score and find a single vector projection
% that maximizes it. If the distribution along that maximal projection crosses a
% bimodality threshold, then the cluster is split along that direction
% it only uses the PC features for each spike, stored in rez.cProjPC

ops = rez.ops;

wPCA = gather(ops.wPCA); % use PCA projections to reconstruct templates when we do splits

ccsplit = rez.ops.AUCsplit; % this is the threshold for splits, and is one of the main parameters users can change

NchanNear   = min(ops.Nchan, 32);
Nnearest    = min(ops.Nchan, 32);
sigmaMask   = ops.sigmaMask;

ik = 0;
Nfilt = size(rez.W,2);
nsplits= 0;

[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear); % determine what channels each template lives on

ops.nt0min = getOr(ops, 'nt0min', 20); % the waveforms must be aligned to this sample

[~, iW] = max(abs(rez.dWU(ops.nt0min, :, :)), [], 2); % find the peak abs channel for each template
iW = squeeze(int32(iW));

isplit = 1:Nfilt; % keep track of original cluster for each cluster. starts with all clusters being their own origin.
dt = 1/1000;
nccg = 0;

while ik<Nfilt
    if rem(ik, 100)==1
      % periodically write updates
       fprintf('Found %d splits, checked %d/%d clusters, nccg %d \n', nsplits, ik, Nfilt, nccg)
    end
    ik = ik+1;

    %
    isp = find(rez.st3(:,2)==ik); % get all spikes from this cluster
    nSpikes = numel(isp);
    if  nSpikes<300
       continue; % do not split if fewer than 300 spikes (we cannot estimate cross-correlograms accurately)
    end

    ss = rez.st3(isp,1)/ops.fs; % convert to seconds

    clp0 = rez.cProjPC(isp, :, :); % get the PC projections for these spikes
    clp0 = gpuArray(clp0(:,:));
    clp = clp0 - mean(clp0,1); % mean center them

    clp = clp - my_conv2(clp, 250, 1); % subtract a running average, because the projections are NOT drift corrected

    % now use two different ways to initialize the bimodal direction
    % the main script calls this function twice, and does both initializations
    if flag
        [u s v] = svdecon(clp');
        w = u(:,1); % initialize with the top PC
    else
        w = mean(clp0, 1)'; % initialize with the mean of NOT drift-corrected trace
        w = w/sum(w.^2)^.5; % unit-normalize
    end

    % initial projections of waveform PCs onto 1D vector
    x = gather(clp * w);
    s1 = var(x(x>mean(x))); % initialize estimates of variance for the first
    s2 = var(x(x<mean(x))); % and second gaussian in the mixture of 1D gaussians

    mu1 = mean(x(x>mean(x))); % initialize the means as well
    mu2 = mean(x(x<mean(x)));
    p  = mean(x>mean(x)); % and the probability that a spike is assigned to the first Gaussian

    logp = zeros(numel(isp), 2); % initialize matrix of log probabilities that each spike is assigned to the first or second cluster

    % do 50 pursuit iteration
    for k = 1:50
        % for each spike, estimate its probability to come from either Gaussian cluster
        logp(:,1) = -1/2*log(s1) - (x-mu1).^2/(2*s1) + log(p);
        logp(:,2) = -1/2*log(s2) - (x-mu2).^2/(2*s2) + log(1-p);

        lMax = max(logp,[],2);
        logp = logp - lMax; % subtract the max for floating point accuracy
        rs = exp(logp); % exponentiate the probabilities

        pval = log(sum(rs,2)) + lMax; % get the normalizer and add back the max
        logP(k) = mean(pval); % this is the cost function: we can monitor its increase

        rs = rs./sum(rs,2); % normalize so that probabilities sum to 1

        p = mean(rs(:,1)); % mean probability to be assigned to Gaussian 1
        mu1 = (rs(:,1)' * x )/sum(rs(:,1)); % new estimate of mean of cluster 1 (weighted by "responsibilities")
        mu2 = (rs(:,2)' * x )/sum(rs(:,2)); % new estimate of mean of cluster 2 (weighted by "responsibilities")

        s1 = (rs(:,1)' * (x-mu1).^2 )/sum(rs(:,1)); % new estimates of variances
        s2 = (rs(:,2)' * (x-mu2).^2 )/sum(rs(:,2));

        if (k>10 && rem(k,2)==1)
            % starting at iteration 10, we start re-estimating the pursuit direction
            % that is, given the Gaussian cluster assignments, and the mean and variances,
            % we re-estimate w
            StS  = clp' * (clp .* (rs(:,1)/s1 + rs(:,2)/s2))/nSpikes; % these equations follow from the model
            StMu = clp' * (rs(:,1)*mu1/s1 + rs(:,2)*mu2/s2)/nSpikes;

            w = StMu'/StS; % this is the new estimate of the best pursuit direection
            w = normc(w'); % which we unit normalize
            x = gather(clp * w);  % the new projections of the data onto this direction
        end
    end

    ilow = rs(:,1)>rs(:,2); % these spikes are assigned to cluster 1
%     ps = mean(rs(:,1));
    plow = mean(rs(ilow,1)); % the mean probability of spikes assigned to cluster 1
    phigh = mean(rs(~ilow,2)); % same for cluster 2
    nremove = min(mean(ilow), mean(~ilow)); % the smallest cluster has this proportion of all spikes


    % did this split fix the autocorrelograms?
    [K, Qi, Q00, Q01, rir] = ccg(ss(ilow), ss(~ilow), 500, dt); % compute the cross-correlogram between spikes in the putative new clusters
    Q12 = min(Qi/max(Q00, Q01)); % refractoriness metric 1
    R = min(rir); % refractoriness metric 2

    % if the CCG has a dip, don't do the split.
    % These thresholds are consistent with the ones from merges.
    if Q12<.25 && R<.05 % if both metrics are below threshold.
        nccg = nccg+1; % keep track of how many splits were voided by the CCG criterion
        continue;
    end

    % now decide if the split would result in waveforms that are too similar
    c1  = wPCA * reshape(mean(clp0(ilow,:),1), 3, []); %  the reconstructed mean waveforms for putatiev cluster 1
    c2  = wPCA * reshape(mean(clp0(~ilow,:),1), 3, []); %  the reconstructed mean waveforms for putative cluster 2
    cc = corrcoef(c1, c2); % correlation of mean waveforms
    n1 =sqrt(sum(c1(:).^2)); % the amplitude estimate 1
    n2 =sqrt(sum(c2(:).^2)); % the amplitude estimate 2

    r0 = 2*abs(n1 - n2)/(n1 + n2); % similarity of amplitudes

    % if the templates are correlated, and their amplitudes are similar, stop the split!!!
    if cc(1,2)>.9 && r0<.2
        continue;
    end

    % finaly criteria to continue with the split: if the split piece is more than 5% of all spikes,
    % if the split piece is more than 300 spikes, and if the confidences for assigning spikes to
    % both clusters exceeds a preset criterion ccsplit
    if nremove > .05 && min(plow,phigh)>ccsplit && min(sum(ilow), sum(~ilow))>300
       % one cluster stays, one goes
       Nfilt = Nfilt + 1;

       % the templates for the splits have been estimated from PC coefficients
       rez.dWU(:,iC(:, iW(ik)),Nfilt) = c2;
       rez.dWU(:,iC(:, iW(ik)),ik)    = c1;

       % the temporal components are therefore just the PC waveforms
       rez.W(:,Nfilt,:) = permute(wPCA, [1 3 2]);
       iW(Nfilt) = iW(ik); % copy the best channel from the original template
       isplit(Nfilt) = isplit(ik); % copy the provenance index to keep track of splits

       rez.st3(isp(ilow), 2)    = Nfilt; % overwrite spike indices with the new index
       rez.simScore(:, Nfilt)   = rez.simScore(:, ik); % copy similarity scores from the original
       rez.simScore(Nfilt, :)   = rez.simScore(ik, :); % copy similarity scores from the original
       rez.simScore(ik, Nfilt) = 1; % set the similarity with original to 1
       rez.simScore(Nfilt, ik) = 1; % set the similarity with original to 1

       rez.iNeigh(:, Nfilt)     = rez.iNeigh(:, ik); % copy neighbor template list from the original
       rez.iNeighPC(:, Nfilt)     = rez.iNeighPC(:, ik); % copy neighbor channel list from the original

       % try this cluster again
       ik = ik-1; % the cluster piece that stays at this index needs to be tested for splits again before proceeding
       % the piece that became a new cluster will be tested again when we get to the end of the list
       nsplits = nsplits + 1; % keep track of how many splits we did

    end
end

fprintf('Finished splitting. Found %d splits, checked %d/%d clusters, nccg %d \n', nsplits, ik, Nfilt, nccg)


Nfilt = size(rez.W,2); % new number of templates
Nrank = 3;
Nchan = ops.Nchan;
Params     = double([0 Nfilt 0 0 size(rez.W,1) Nnearest ...
    Nrank 0 0 Nchan NchanNear ops.nt0min 0]); % make a new Params to pass on parameters to CUDA

% we need to re-estimate the spatial profiles
[Ka, Kb] = getKernels(ops, 10, 1); % we get the time upsampling kernels again
[rez.W, rez.U, rez.mu] = mexSVDsmall2(Params, rez.dWU, rez.W, iC-1, iW-1, Ka, Kb); % we run SVD

[WtW, iList] = getMeWtW(single(rez.W), single(rez.U), Nnearest); % we re-compute similarity scores between templates
rez.iList = iList; % over-write the list of nearest templates

isplit = rez.simScore==1; % overwrite the similarity scores of clusters with same parent
rez.simScore = gather(max(WtW, [], 3));
rez.simScore(isplit) = 1; % 1 means they come from the same parent

rez.iNeigh   = gather(iList(:, 1:Nfilt)); % get the new neighbor templates
rez.iNeighPC    = gather(iC(:, iW(1:Nfilt))); % get the new neighbor channels

rez.Wphy = cat(1, zeros(1+ops.nt0min, Nfilt, Nrank), rez.W); % for Phy, we need to pad the spikes with zeros so the spikes are aligned to the center of the window

rez.isplit = isplit; % keep track of origins for each cluster


% figure(1)
% subplot(1,4,1)
% plot(logP(1:k))
%
% subplot(1,4,2)
% [~, isort] = sort(x);
% epval = exp(pval);
% epval = epval/sum(epval);
% plot(x(isort), epval(isort))
%
% subplot(1,4,3)
% ts = linspace(min(x), max(x), 200);
% xbin = hist(x, ts);
% xbin = xbin/sum(xbin);
%
% plot(ts, xbin)
%
% figure(2)
% plotmatrix(v(:,1:4), '.')
%
% drawnow
%
% % compute scores for splits
% ilow = rs(:,1)>rs(:,2);
% ps = mean(rs(:,1));
% [mean(rs(ilow,1)) mean(rs(~ilow,2)) max(ps, 1-ps) min(mean(ilow), mean(~ilow))]
