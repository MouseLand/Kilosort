function [imin,yblk, F0] = align_block2(F, ysamp)

Nbatches = size(F,3);
n = 15;
dc = zeros(2*n+1, Nbatches);
dt = -n:n;

Fg = gpuArray(single(F));

Fg = Fg - mean(Fg, 1);
F0 = Fg(:,:, min(300, floor(size(Fg,3)/2)));

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

nybins = size(F,1);
nblocks = 6;
yl = floor(nybins/nblocks);
ifirst = round(linspace(1, nybins - yl, 2*nblocks-1));
ilast  = ifirst + yl; %287;

%%

nblocks = length(ifirst);
yblk = zeros(length(ifirst), 1);

n = 5;
dt = -n:n;

dcs = zeros(2*n+1, Nbatches, nblocks);
for j = 1:nblocks
    isub = ifirst(j):ilast(j);
    yblk(j) = mean(ysamp(isub));
    
%     Fsub = gpuArray.zeros(length(isub) + 2*n, 20, Nbatches);
%     Fsub(isub +n, :, :) = Fg(isub, :, :) - mean(Fg(isub, :, :), 1);
     Fsub = Fg(isub, :, :);
     
    for t = 1:length(dt)
        Fs = circshift(Fsub, dt(t), 1);
        dcs(t, :, j) = gather(sq(mean(mean(Fs .* F0(isub, :, :), 1), 2)));
    end
end
%%
dtup = linspace(-n, n, (2*n*10)+1);    
K = kernelD(dt,dtup,1);
dcs = my_conv2(dcs, .5, [1, 2, 3]);

imin = zeros(Nbatches, nblocks);
for j = 1:nblocks
    dcup = K' * dcs(:,:,j);
    
    [~, imax] = max(dcup, [], 1);
    dall(niter, :) = dtup(imax);
    
    imin(:,j) = sum(dall,1);
end

% dtup = linspace(-n, n, (2*n*10)+1);
% 
% K = kernelD(dt,dtup,1);
% dcup = K' * dc;
% [~, imax] = max(dcup, [], 1);
% dall(niter, :) = dtup(imax);
% 
% imin = sum(dall,1);