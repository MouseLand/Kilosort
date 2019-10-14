function Wrot = whiteningLocal(CC, yc, xc, nRange)
% function to perform local whitening of channels
% CC is a matrix of Nchan by Nchan correlations
% yc and xc are vector of Y and X positions of each channel
% nRange is the number of nearest channels to consider
Wrot = zeros(size(CC,1), size(CC,1));
for j = 1:size(CC,1)
    ds          = (xc - xc(j)).^2 + (yc - yc(j)).^2;
    [~, ilocal] = sort(ds, 'ascend');
    ilocal      = ilocal(1:nRange); % take the closest channels to the primary channel. First channel in this list will always be the primary channel.

    wrot0 = whiteningFromCovariance(CC(ilocal, ilocal));
    Wrot(ilocal, j)  = wrot0(:,1); % the first column of wrot0 is the whitening filter for the primary channel
end
