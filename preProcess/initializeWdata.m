function [W, mu, Wheights] = initializeWdata(ioff, uprojDAT, nCHmax, nPCs, Nfilt)

[nFeat nS]       = size(uprojDAT); % number of spikes

W = gpuArray.zeros(nCHmax*nPCs, Nfilt, 'single');
for j = 1:size(W,2)
   W(double(ioff(j)) + [1:nFeat], j) = uprojDAT(:, j);
end

mu = sum(W.^2,1).^.5;
W = W./(1e-5 + mu);

W = reshape(W, nPCs, nCHmax, Nfilt);
nW = sq(sum(W(1, :, :).^2,1));
W = reshape(W, nPCs* nCHmax, Nfilt);

[~, Wheights] = max(nW,[], 1);
[Wheights, isort] = sort(Wheights);

W = W(:, isort);
