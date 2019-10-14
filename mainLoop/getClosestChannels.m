function [iC, mask, C2C] = getClosestChannels(rez, sigma, NchanClosest)
% this function outputs the closest channels to each channel,
% as well as a Gaussian-decaying mask as a function of pairwise distances
% sigma is the standard deviation of this Gaussian-mask

% compute distances between all pairs of channels
C2C = (rez.xc(:) - rez.xc(:)').^2 + (rez.yc(:) - rez.yc(:)').^2;
C2C =  sqrt(C2C);
Nchan = size(C2C,1);

% sort distances
[~, isort] = sort(C2C, 'ascend');

% take NchanCLosest neighbors for each primary channel
iC= isort(1:NchanClosest, :);

% in some cases we want a mask that decays as a function of distance between pairs of channels
ix = iC + [0:Nchan:Nchan^2-1]; % this is an awkward indexing to get the corresponding distances
mask = exp( - C2C(ix).^2/(2*sigma^2));

mask = mask ./ (1e-3 + sum(mask.^2,1)).^.5; % masks should be unit norm for each channel


iC = gpuArray(int32(iC)); % iC and mask live on the GPU
mask = gpuArray(single(mask));
