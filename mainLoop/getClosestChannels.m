function [iC, mask] = getClosestChannels(rez, sigma, NchanClosest)


pds = (rez.xc(:) - rez.xc(:)').^2 + (rez.yc(:) - rez.yc(:)').^2;

Nchan = size(pds,1);

pds =  sqrt(pds);

[~, isort] = sort(pds, 'ascend');

iC= isort(1:NchanClosest, :);
iC = sort(iC, 1);

ix = iC + [0:Nchan:Nchan^2-1];
mask = exp( - pds(ix).^2/(2*sigma^2));

mask = mask ./ (1e-3 + sum(mask.^2,1)).^.5;


iC = gpuArray(int32(iC));
mask = gpuArray(single(mask));

