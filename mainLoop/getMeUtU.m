function [UtU, maskU, iList] = getMeUtU(iU, iC, mask, Nnearest, Nchan)

Nfilt = numel(iU);

U = gpuArray.zeros(Nchan, Nfilt, 'single');

ix = iC(:, iU) + int32([0:Nchan:(Nchan*Nfilt-1)]);
U(ix) = 1;

UtU = (U'*U) > 0;

maskU = mask(:, iU);

if nargin>3 && nargout>2
    cc = UtU;
    [~, isort] = sort(cc, 1, 'descend');
    iList = int32(gpuArray(isort(1:Nnearest, :)));
end