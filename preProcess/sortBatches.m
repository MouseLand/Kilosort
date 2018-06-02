function [ccb0, isort] = sortBatches(ccb0)

ccb0 = gpuArray(ccb0);

nBatches = size(ccb0,1);

[u, s, v] = svdecon(ccb0);

u = u - mean(u,1);
theta = angle(u(:,1) + 1i*u(:,2));

theta = gpuArray(theta);

A0 = gpuArray.ones(nBatches, 'single');
eta = .25;

niB = 100;
Cost = zeros(1, niB);

for k = 1:niB
    A1 = sin(theta) * sin(theta');
    A2 = cos(theta) * cos(theta');
    
    x = [ A1(:) A2(:) A0(:)];
    y = ccb0(:);
    
    B = (x'*x)\(x'*y);
    ypred = x * B;
    
    err = ypred - y;
    
    Cost(k) = gather(mean(err(:).^2));
    err = reshape(err, nBatches, nBatches);
    
    d1 = B(1) * (err * sin(theta)) .* cos(theta);
    d2 = - B(2) * (err * cos(theta)) .* sin(theta);
    
    dTheta = d1 + d2;
    
    dTheta = dTheta/nBatches;
    theta = theta - eta * dTheta;
end

% figure
% plot(Cost(1:niB))

theta = mod(theta, 2*pi);
thsort = sort(theta);
thsort = [thsort(end)-2*pi; thsort];
[~, imax] = max(diff(thsort));
t0 = thsort(imax); 

ix = theta < t0;
theta(ix) = theta(ix) + 2*pi;

[~, isort] = sort(theta);

ccb0 = ccb0(isort, isort);

% [u, s, v] = svdecon(ccb0);
% [~, isort] = sort(u(:,1));
