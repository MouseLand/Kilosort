function [ccsort, isort, isort2] = sort_by_rastermap(cc)

nC = 50;

cc = cc - diag(diag(cc));
cc = cc + diag(mean(cc,1));
ccsort = zscore(cc, [], 1);
[u, s, v] = svdecon(ccsort);
[~, isort] = sort(u(:,1));
X = u(:, 1:200) * s(1:200, 1:200);


SALL = dctbasis(nC,1);
SALL = SALL(:, 2:end);

[NN, nPC] = size(X);
[~, isort] = sort(u(:,1));

nn = floor(NN/nC);
iclust = zeros(NN, 1);
iclust(isort) = ceil([1:NN]/nn);
iclust(iclust>nC) = nC;

niter = 50;
nBasis = size(SALL, 2);
ncomps = linspace(3, nBasis, niter-20);
ncomps = cat(2, ncomps, linspace(nBasis, nBasis, 20));
ncomps = int16(ncomps);

lam = ones(NN,1);
alpha = 3;
f = 1 ./ (1 + [1:nBasis]).^(alpha/2);
xnorm = sum(X.^2, 2);
for j = 1:length(ncomps)
    nc = ncomps(j);
    S = SALL(:, 1:nc);
    X0 = zeros(nPC, nC);
    for k = 1:nC
        ix = iclust==k;
        if sum(ix)>-7
            lam(ix) = lam(ix) / sum(lam(ix).^2)^.5;
            X0(:, k) = lam(ix)' * X(ix, :);
        end
    end
    
    A = X0 * S;
    nA = 1e-10 + sum(A.^2, 1).^.5;
    nA = nA ./ f(1:nc);
    
    eweights = ((S ./ nA) * S');
    eweights = eweights(iclust, :) .*  lam;
    
    A = A ./ nA;
    
    AtS = A * S';
    
    vnorm = sum(AtS.^2, 1);
    cv = X * AtS;
    
    vnorm = vnorm + xnorm .* eweights.^2 - 2 * eweights .* cv;
    cv = cv - xnorm .* eweights;
    
    
    cmap = max(0, cv).^2./vnorm;
    [cmax, iclust] = max(cmap, [], 2);
    
    lam = cmax ./ vnorm((iclust-1)*NN + [1:NN]');
    lam= lam.^.5;
    
    ce(j) = sum(cmax)/sum(xnorm);
%     if rem(j,10)==1 || j==length(ncomps)
%         disp(ce(j))
%     end
end

[~, isort2] = sort(iclust);
CC      = sqrt(cmap);
eta     = .5;
isort   = upsample_grad(CC, eta, nC);

ccsort = ccsort(isort, isort);
end

function inewsort = upsample_grad(CC, eta, nC)
CC = CC ./ max(CC, [], 2);
[~, xid] = max(CC, [], 2);
y0 = 1:nC;
sig = 1;
niter = 201;
yinit = y0(xid);
y = yinit;
eta = linspace(eta, eta/10, niter);
for j =1:niter
    yy0 = y' - y0;
    K = exp(-yy0.^2/(2*sig^2));
    x = mean(mean(CC .* K)) /mean(mean(K.^2));
    err = x * K - CC;
    Kprime = - x * yy0.*K;
    dy = sum(Kprime .* err, 2)';
    y = y - eta(j) * dy;
    E(j) = mean(err(:).^2);
end

[~, inewsort] = sort(y);
end
