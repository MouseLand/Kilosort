function [iC, dist] = getClosestChannels2(ycup, xcup, yc, xc, NchanClosest)
% this function outputs the closest channels to each channel,
% as well as a Gaussian-decaying mask as a function of pairwise distances
% sigma is the standard deviation of this Gaussian-mask

% compute distances between all pairs of channels
C2C = (xc(:) - xcup(:)').^2 + (yc(:) - ycup(:)').^2;
C2C =  sqrt(C2C);
[Nchan, NchanUp] = size(C2C);

% sort distances
[~, isort] = sort(C2C, 1, 'ascend');

% take NchanCLosest neighbors for each primary channel
iC= isort(1:NchanClosest, :);

% in some cases we want a mask that decays as a function of distance between pairs of channels
ix = iC + [0:Nchan:Nchan*NchanUp-1]; % this is an awkward indexing to get the corresponding distances

dist = gpuArray(single(C2C(ix)));
iC = gpuArray(int32(iC)); % iC and mask live on the GPU

