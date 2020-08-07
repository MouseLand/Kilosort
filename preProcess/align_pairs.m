function d0 = align_pairs(F1, F2)  

% this little function aligns one average "frame" to another "frame" for
% chronio alignment

n = 15;
dc = zeros(2*n+1, 1);
dt = -n:n;

% do the loop
for t = 1:length(dt)
    % use one frame as the reference, and compute correlations for each
    % potential shift 
    Fs = circshift(F2, dt(t), 1);
    dc(t) = gather(sq(mean(mean(F1 .* F2, 1), 2)));
end

dtup = linspace(-n, n, (2*n*10)+1);

% upsample the correlation curves
K = kernelD(dt,dtup,1);
dcup = K' * dc;
[~, imax] = max(dcup, [], 1);
d0 = dtup(imax);
