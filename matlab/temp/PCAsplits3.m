% 
Nfilt = size(rez.W,2);
% ik = ceil(rand * Nfilt);
% ik = 1;

sp = find(rez.st3(:,2)==ik);
sp = sort(sp);

st = rez.st3(sp, 1);

clp = rez.cProjPC(sp, :, :);
clp = clp - mean(clp,1);

clp = gpuArray(clp(:,:));

[u s v] = svdecon(clp(:,:)');

v = s * v';

v = v(1:2, :)';

npivots = min( size(v,1), 1e3); %size(v,1);

nspikes = size(v,1);

ipick = randperm(nspikes, npivots);

cn = sum(v.^2,2);

tic
ds = v * v(ipick, :)';
ds = -2*ds + cn + cn(ipick)';
ds = sqrt(max(0, ds));
toc

dns = mean(ds<2);

p2 = ds(ipick, :);

d2s = (dns - dns') .* (p2<3);

[amax, imax] = max(d2s, [], 2);

in = find(amax<1e-10);

numel(in)


figure(100)
plot(v(:, 1), v(:,2), '.')
hold on
plot(v(ipick(in), 1), v(ipick(in),2), 'or', 'MarkerFaceColor', 'r', 'Markersize', 4)
hold off


%%
x1 = v(ipick(in(1)), :);
x2 = v(ipick(in(2)), :);

Y = v * (x1-x2)';


figure(100)
hist(Y, 100)

figure(1)
% close all
plotmatrix(v(:,1:4), '.')


