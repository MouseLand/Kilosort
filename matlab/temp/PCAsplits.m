% 
Nfilt = size(rez.W,2);
ik = ceil(rand * Nfilt);
% ik = 1;
%
sp = find(rez.st3(:,2)==ik);
sp = sort(sp);

st = rez.st3(sp, 1);

clp = rez.cProjPC(sp, :, :);
clp = clp - mean(clp,1);
% clp = clp - my_conv2(clp, 50, 1);

nspikes = size(clp,1);
[u s v] = svdecon(clp(:,:)');

figure(1)

% plot(sp, v(:,1), '.')


% close all
plotmatrix(v(:,1:4), '.')


figure(2)
vplot = clp(:,:) * u(:,1:2);
plot(st, my_conv2(vplot(:,1:2), 1, 1))

figure(3)
clp = clp - my_conv2(clp, 100, 1);

nspikes = size(clp,1);
[u s v] = svdecon(clp(:,:)');

plotmatrix(v(:,1:4), '.')

clp = rez.cProjPC(sp, :, :);
clp = clp - mean(clp,1);

nspikes = size(clp,1);
[u s v] = svdecon(clp(:,:)');


nPC = min(size(v,2), round(size(v,1)^.5));

X = v(:, 1:nPC) * size(v,1)^.5;
[numSamples, ncoefs] = size(X);

kPC = kurtosis(v, [], 1);
% [~, imin] = max(abs(kPC-3));
[~, imin] = min(kPC(1:min(4,nPC)));
[nPC imin]
%
B = eye(ncoefs, 4);

eta = .05;
p = 0;
gp = zeros(size(B));

figure(4)
clear C
for k = 1:30
    Y = X * B;
%     
%     g = X' * (Y .^ 3) / numSamples;        
%     gp = p * gp + (1-p) * g;    
%     B = B - eta * gp;
    
    B = X' * (Y .^ 3) / numSamples - 3*B;         
    B = B * real(inv(B' * B)^(1/2));
    
    B = normc(B);
    
    C(k) = mean(Y(:).^4);
end
for i = 1:4
    subplot(2,2,i)
    hist(Y(:,i), 100)
end
%%

hist(Y, 100)