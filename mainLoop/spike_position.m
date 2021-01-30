function xy = spike_position(dd, wPCA, wTEMP, yc, xc)

wTPC = gpuArray(wPCA' * wTEMP);

dT = gpuArray(dd);
for j = 1:size(dd,3)
    dT(:,:,j) = dT(:,:,j) * wTPC;
end

[nspikes, nPC, nchan] = size(dT);
[~, imax] = max(max(dT.^2, [], 3), [], 2);
dBest = gpuArray.zeros(nspikes, nchan, 'single');
for j = 1:nPC
    iX = imax==j;
    dBest(iX, :) = dT(iX, j, :);
end

dBest = max(0, dBest);
dBest = dBest ./ sum(dBest,2);

ysp = dBest * yc;
xsp = dBest * xc;

xy = [xsp, ysp];

xy = gather(xy);
end