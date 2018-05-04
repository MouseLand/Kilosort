function [d2d, iY, drez] = distance_betwxt(dWU)
[nt0, Nchan, Nfilt] = size(dWU);

dWU = reshape(dWU, nt0*Nchan, Nfilt);
d2d = dWU' * dWU;

mu = sum(dWU.^2,1).^.5;
mu = mu';

muall2 = repmat(mu.^2, 1, Nfilt);
d2d = 1 - 2 * d2d./(1e-30 + muall2+ muall2');

d2d  = 1- triu(1 - d2d, 1);

[dMin, iY] = min(d2d, [], 1);

drez = dMin;


end

