function [W, U, dWU, mu, nsp, his, flag] = triageHIST(ops, W, U, dWU, mu, nsp, his, flag)

Nfilt = size(W,2);

[mh, imax, his1, his2] = split_histograms(gather(his));
% 
% figure(3)
% plot(mh)
% drawnow

isplit = find((mh > ops.splitT) & nsp'>.1);
% isplit = find((mh > ops.splitT) & flag>10 & nsp'>2);
n_new = numel(isplit);

if n_new>0
    fac = linspace(.5, 1.5, 101);
    fac = fac(1:100);
    
    f1 = (fac * his1(:, isplit))./sum(his1(:, isplit),1);
    % f2 = (fac * his2(:, isplit))./sum(his2(:, isplit),1);
    
    inew = Nfilt + [1:n_new];
    
    W(:,inew, :) = W(:, isplit, :);
    U(:,inew, :) = U(:, isplit, :);
    
    mu(inew)   = mu(isplit) .* f1';
    % mu(isplit) = mu(isplit) .* f1';
    
    dWU(:,:,inew)   = dWU(:, :, isplit) .* permute(f1, [3 1 2]);
    % dWU(:,:,isplit) = dWU(:, :, isplit) .* permute(f1, [3 1 2]);
    
    
    nsp(inew)   = sum(his1(:, isplit),1);
    % nsp(isplit) = sum(his2(:, isplit),1);
    
    % nsp(inew) = nsp(isplit); % this gives the weaker cluster a chance to survive
    
    n0 = exp(- 2*(fac(:) - 1).^2);
    n0 = n0/sum(n0);
    
    his(:,inew)   = n0 * nsp(inew)';
    his(:,isplit) = his2(:, isplit);    
    
    flag(inew) = 0;
    flag(isplit) = 0;
end

flag = flag + 1;