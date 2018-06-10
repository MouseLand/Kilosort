function [Ka, Kb] = getKernels(ops, nup, sig)
% nup = 10;

nt0max = getOr(ops, 'nt0max', 20);
nt0    = getOr(ops, 'nt0',    61);

xs = 1:nt0;
ys = linspace(.5, nt0+.5, nt0*nup+1);
ys(end) = [];

% sig  = .5;

d = rem(xs(:) - xs(:)' + nt0, nt0);
d = min(d, nt0-d);
Kxx = exp(-d.^2 / (sig^2));

d = rem(ys(:) - xs(:)' + nt0, nt0);
d = min(d, nt0-d);
Kyx = exp(-d.^2 / (sig^2));

B = Kyx/(Kxx + .01 * eye(nt0));
B = reshape(B, nup, nt0, nt0);

A = squeeze(B(:, nt0max, :));
B = permute(B, [2 3 1]);

Ka = gpuArray(single(A));
Kb = gpuArray(single(B));

% A = ;

