function [row, col, mu] = isolated_peaks(S1, ops)
loc_range = ops.loc_range;
long_range = ops.long_range; 
Th = ops.spkTh;
nt0 = ops.nt0;

% loc_range = [3  1];
% long_range = [30  6];
smin = my_min(S1, loc_range, [1 2]);
peaks = single(S1<smin+1e-3 & S1<Th);

sum_peaks = my_sum(peaks, long_range, [1 2]);
peaks = peaks .* (sum_peaks<1.2) .* S1;

% exclude temporal buffers
peaks([1:nt0 end-nt0:end], :) = 0;

% exclude edge channels 
noff = 8;
peaks(:, [1:noff end-noff+ [1:noff]]) = 0;

[row, col, mu] = find(peaks);

mu = - mu;
