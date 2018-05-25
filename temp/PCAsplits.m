% 
Nfilt = size(rez.W,2);
ik = ceil(rand * Nfilt);
ik = 213;
%
sp = find(rez.st3(:,2)==ik);
sp = sort(sp);

st = rez.st3(sp, 1);

clp = rez.cProjPC(sp, :, :);

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

figure(4)

% plot(muall(ik, :))
%%

imagesc(sq(Wall(:, ik, 1, :)))

%%

