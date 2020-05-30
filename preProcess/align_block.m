function [imin, F0] = align_block(F)

Nbatches = size(F,3);
n = 15;
dc = zeros(2*n+1, Nbatches);
dt = -n:n;

Fg = gpuArray(single(F));

Fg = Fg - mean(Fg, 1);
F0 = Fg(:,:, 300);

% do the loop
niter = 10;
dall = zeros(niter, Nbatches);
for iter = 1:niter
    for t = 1:length(dt)
        Fs = circshift(Fg, dt(t), 1);
        dc(t, :) = gather(sq(mean(mean(Fs .* F0, 1), 2)));
    end
    
    if iter<niter
        [~, imax] = max(dc, [], 1);
        
        for t = 1:length(dt)
            ib = imax==t;
            Fg(:, :, ib) = circshift(Fg(:, :, ib), dt(t), 1);
            dall(iter, ib) = dt(t);
        end
        
        F0 = mean(Fg, 3);
    end
end

%%
dtup = linspace(-n, n, (2*n*10)+1);

K = kernelD(dt,dtup,1);
dcup = K' * dc;
[~, imax] = max(dcup, [], 1);
dall(niter, :) = dtup(imax);

imin = sum(dall,1);