% 
ik = ceil(rand * Nfilt);


subplot(1,2,1)
clp = fWpc(:,:,rez.st3(:,2)==ik);
nspikes = size(clp,3);
[u s v] = svdecon(reshape(clp, [], nspikes));
plotmatrix(v(:,1:4), v(:,1:4), '.')


subplot(1,2,2)

clp = fWpc0(:,:,rez.st3(:,2)==ik);
nspikes = size(clp,3);
[u s v] = svdecon(reshape(clp, [], nspikes));

plotmatrix(v(:,1:4), v(:,1:4), '.')
%%
figure(2)

plot(u(:,1:2))