function WtW = pairwise_dists(WU, WUinit)

WU = reshape(WU, [], size(WU,3));
WUinit = reshape(WUinit, [], size(WUinit,3));
WtW = WU' * WUinit;


mu = sum(WU.^2,1);
mu = mu(:);

muinit = sum(WUinit.^2,1);
muinit = muinit(:);

mu     = repmat(mu, 1, size(WUinit,2));
muinit = repmat(muinit', size(WU,2), 1);

WtW = 1 - 2*WtW ./ (muinit + mu);