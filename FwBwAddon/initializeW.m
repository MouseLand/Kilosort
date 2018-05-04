function [W, Wheights] = initializeW(Nfilt, nCHmax, nPCs)
W = gpuArray.zeros(7, nCHmax, Nfilt, 'single');
ic = round(linspace(1, nCHmax, Nfilt));
W(7*(ic-1)+1 + 7 * nCHmax * [0:1:Nfilt-1]) = -100;
W = reshape(W, nCHmax * nPCs, Nfilt);
W = W + 1 * gpuArray.randn(nCHmax * nPCs, Nfilt, 'single');
W = normc(W);
W = reshape(W, nPCs, nCHmax, Nfilt);
nW = sq(sum(W(1, :, :).^2,1));
[~, Wheights] = max(nW,[], 1);
[~, isort] = sort(Wheights);
W = reshape(W, nCHmax * nPCs, Nfilt);
W = W(:, isort);
end