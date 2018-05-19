function [W0, nmax] = get_diffW(W)


W0 = gpuArray.zeros(size(W), 'single');

W0(2:end-1, :, :) = (W(3:end, :, :) - W(1:end-2, :, :))/2;
W0  = W0 - W .* sum(W0.*W,1);
nmax = sum(W0.^2,1).^.5;
W0 = normc(W0);
