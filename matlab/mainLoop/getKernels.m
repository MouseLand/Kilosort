function [Ka, Kb] = getKernels(ops, nup, sig)
% this function makes upsampling kernels for the temporal components.
% those are used for interpolating the biggest negative peak,
% and aligning the template to that peak with sub-sample resolution
% needs nup, the interpolation factor (default = 10)
% also needs sig, the interpolation smoothness (default = 1)

nt0min = getOr(ops, 'nt0min', 20);
nt0    = getOr(ops, 'nt0',    61);

xs = 1:nt0;
ys = linspace(.5, nt0+.5, nt0*nup+1);
ys(end) = [];

% these kernels are just standard kriging interpolators

% first compute distances between the sample coordinates
% for some reason, this seems to be circular, although the waveforms are not circular
% I think the reason had to do with some constant offsets in some channels?
d = rem(xs(:) - xs(:)' + nt0, nt0);
d = min(d, nt0-d);
Kxx = exp(-d.^2 / (sig^2)); % the kernel covariance uses a squared exponential of spatial scale sig

% do the same for the kernel similarities between upsampled "test" timepoints and the original coordinates
d = rem(ys(:) - xs(:)' + nt0, nt0);
d = min(d, nt0-d);
Kyx = exp(-d.^2 / (sig^2));

% the upsampling matrix is given by the following formula,
% with some light diagonal regularization of the matrix inversion
B = Kyx/(Kxx + .01 * eye(nt0));
B = reshape(B, nup, nt0, nt0);

% A is just a slice through this upsampling matrix corresponding to the most negative point
% this is used to compute the biggest negative deflection (after upsampling)
A = squeeze(B(:, nt0min, :));
B = permute(B, [2 3 1]);

% move to the GPU and make it a double
Ka = gpuArray(double(A));
Kb = gpuArray(double(B));
