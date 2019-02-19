function [W, U, dWU, mu, nsp,sig, dnext, damp, ndrop] = ...
    triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp,sig, dnext, damp, ndrop)

m0 = ops.minFR * ops.NT/ops.fs;
idrop = nsp<m0;

W(:,idrop,:) = [];
U(:,idrop,:) = [];
dWU(:,:, idrop) = [];
mu(idrop) = [];
nsp(idrop) = [];
sig(idrop) = [];
dnext(idrop) = [];
ndrop(1) = .9 * ndrop(1) + .1*gather(sum(idrop));

% 
cc = getMeWtW2(W, U);
cc = cc -diag(diag(cc));
sd = sqrt(sig);
r0 = 2*(sd(:) + sd(:)') ./ abs(mu(:) - mu(:)');
rdir = (nsp(:) - nsp(:)')<0;
ipair = (cc>0.9 & r0>1 & rdir);
amax = max(ipair, [], 2);
idrop= amax>0;

W(:,idrop,:) = [];
U(:,idrop,:) = [];
dWU(:,:, idrop) = [];
mu(idrop) = [];
nsp(idrop) = [];
sig(idrop) = [];
dnext(idrop) = [];
ndrop(2) = .9*ndrop(2) + .1*gather(sum(idrop));

% check which templates can be absorbed into other templates

mergeThreshold = getOr(ops, 'mergeThreshold', 0);

imerge = find(sqrt(dnext(:)) < mergeThreshold * mu);
iW0 = iW(imerge);
nsp0 = nsp(imerge);

p2p = (C2C(iW0, iW0)<60) .* (nsp0(:) > nsp0(:)');
imax = max(p2p, [], 2) < .5;

idrop = false(numel(mu), 1);
idrop(imerge(imax)) = 1;

W(:,idrop,:) = [];
U(:,idrop,:) = [];
dWU(:,:, idrop) = [];
mu(idrop) = [];
nsp(idrop) = [];
sig(idrop) = [];
dnext(idrop) = [];

damp(idrop, :) = [];
ndrop(3) = .9*ndrop(3) + .1*gather(sum(idrop));

