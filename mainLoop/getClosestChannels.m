function [iC, mask, C2C] = getClosestChannels(rez, sigma, NchanClosest)


C2C = (rez.xc(:) - rez.xc(:)').^2 + (rez.yc(:) - rez.yc(:)').^2;
Nchan = size(C2C,1);

C2C =  sqrt(C2C);

[~, isort] = sort(C2C, 'ascend');

iC= isort(1:NchanClosest, :);
% iC = sort(iC, 1);

ix = iC + [0:Nchan:Nchan^2-1];
mask = exp( - C2C(ix).^2/(2*sigma^2));

mask = mask ./ (1e-3 + sum(mask.^2,1)).^.5;


iC = gpuArray(int32(iC));
mask = gpuArray(single(mask));

