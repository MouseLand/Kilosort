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

plotmatrix(v(:,1:4), '.')


figure(2)
vplot = clp(:,:) * u(:,1:2);
plot(st, my_conv2(vplot(:,1:2), 1, 1))


figure(3)
clp = rez.cProjPC(sp, :, :);
clp = clp - mean(clp,1);

nspikes = size(clp,1);
[u s v] = svdecon(clp(:,:)');

cls = (clp(:,:) * u(:,1:4)) * u(:,1:4)';

clp = rez.cProjPC(sp, :, :);
clp = clp - mean(clp,1);

for i = 1:16
    subplot(4,4,i)
    l = randperm(size(clp,1), 2);
    w = cls(l(1), :) - cls(l(2), :);
    w = zscore(w);
    
    Y = clp(:,:) * w';
    
    hist(Y, 100)
end
%%

hist(Y, 100)