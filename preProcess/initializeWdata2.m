function [W, mu, Wheights, irand] = initializeWdata2(call, uprojDAT, Nchan, nPCs, Nfilt, iC)

irand = ceil(rand(Nfilt,1) * size(uprojDAT,2));
% irand = 1:Nfilt;

W = gpuArray.zeros(nPCs, Nchan, Nfilt, 'single');

for t = 1:Nfilt
    ich = iC(:, call(irand(t)));
    W(:, ich, t) = reshape(uprojDAT(:, irand(t)), nPCs, []);
end
W = reshape(W, [], Nfilt);
W = W + .001 * gpuArray.randn(size(W), 'single');

mu = sum(W.^2,1).^.5;
W = W./(1e-5 + mu);

W = reshape(W, nPCs, Nchan, Nfilt);
nW = sq(sum(W(1, :, :).^2,1));
W = reshape(W, nPCs * Nchan, Nfilt);

[~, Wheights] = max(nW,[], 1);
