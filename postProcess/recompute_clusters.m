function rez = recompute_clusters(rez)

%recompute cluster statistics after a post-processing operation

NchanNear   = min(rez.ops.Nchan, 32);
Nnearest    = min(rez.ops.Nchan, 32);
iC = getClosestChannels(rez, rez.ops.sigmaMask, NchanNear);
Nfilt = size(rez.W,2); % new number of templates
Nrank = 3;
Nchan = rez.ops.Nchan;
Params     = double([0 Nfilt 0 0 size(rez.W,1) Nnearest ...
    Nrank 0 0 Nchan NchanNear rez.ops.nt0min 0]); % make a new Params to pass on parameters to CUDA

% we need to re-estimate the spatial profiles
[Ka, Kb] = getKernels(rez.ops, 10, 1); % we get the time upsampling kernels again
[~, iW] = max(abs(rez.dWU(rez.ops.nt0min, :, :)), [], 2); % find the peak abs channel for each template
iW = squeeze(int32(iW));
[rez.W, rez.U, rez.mu] = mexSVDsmall2(Params, rez.dWU, rez.W, iC-1, iW-1, Ka, Kb); % we run SVD

[WtW, iList] = getMeWtW(single(rez.W), single(rez.U), Nnearest); % we re-compute similarity scores between templates
rez.iList = iList; % over-write the list of nearest templates

rez.simScore = gather(max(WtW, [], 3));

rez.iNeigh   = gather(iList(:, 1:Nfilt)); % get the new neighbor templates
rez.iNeighPC    = gather(iC(:, iW(1:Nfilt))); % get the new neighbor channels

rez.Wphy = cat(1, zeros(1+rez.ops.nt0min, Nfilt, Nrank), rez.W); % for Phy, we need to pad the spikes with zeros so the spikes are aligned to the center of the window
end