function [ccb1, isort] = sortBatches2(ccb0)

ccb0 = gpuArray(ccb0);

% nBatches = size(ccb0,1);

[u, s, v] = svdecon(ccb0);

xs = .01 * u(:,1)/std(u(:,1));

niB = 200;
% Cost = zeros(1, niB);

eta = 1;
for k = 1:niB
    ds = (xs - xs').^2;
    W = log(1 + ds);
    
    err = ccb0 - W;
    err = err - mean(err(:));
    
%     Cost(k) = gather(mean(err(:).^2));
    
    err = err./(1+ds);
    err2 = err .* (xs - xs');
    D = mean(err2, 2);
    E = mean(err2, 1);
   
    dx = -D + E';
    
    xs = xs - eta * dx;
end

[~, isort] = sort(xs);

ccb1 = ccb0(isort, isort);
