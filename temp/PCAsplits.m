% 
ik = ceil(rand * 200);
ik = 313;

sp = find(rez.st3(:,2)==ik);
sp = sort(sp);

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
plot(my_conv2(vplot(:,1:2), 50, 1))

figure(3)
clp = clp - my_conv2(clp, 50, 1);

nspikes = size(clp,1);
[u s v] = svdecon(clp(:,:)');

plotmatrix(v(:,1:4), '.')
%%

mean(rez.st3(sp(v(:,1)<0), 4))
%%

