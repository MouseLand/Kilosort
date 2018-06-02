function isort = get1Dordering(Ff)
%%

S = zscore(Ff, 1, 2)/size(Ff,2).^.5;
S = gpuArray(single(S));

[U, ~,~] = svdecon(S);
U = U(:,1);
%%
[NN, NT] = size(S);

iclust = zeros(NN, 1);
[~, isort] = sort(U);

nC = 30;
nn = floor(NN/nC);
iclust(isort) = ceil([1:NN]/nn);
iclust(iclust>nC) = nC;

sig = [linspace(3,1,25) 1*ones(1,50)];


V = gpuArray.zeros(NT, nC, 'single');
for j = 1:nC
    ix = iclust== j;
    V(:, j) = mean(S(ix, :),1);
end
C = gpuArray.zeros(nC, NN, 'single');
cv = S * V;
[cmax, iclust] = max(cv, [], 2);
C(iclust + [0:nC:(NN*nC-1)]') = cmax;


for t = 1:numel(sig)
    V = (C * C' + 1e-3 * eye(size(C,1)))\(C * S);
    V = V';
   
   V = my_conv2(V, sig(t), 2);
   V = normc(V);
   
   % multiple coefficients per cell
   S0 = S;
   C = gpuArray.zeros(nC, NN, 'single');
   for k = 1
       cv = S0 * V;
       [cmax, iclust] = max(cv, [], 2);
       
       S0 = S0 - cmax .* V(:, iclust)';
       
       C(iclust + [0:nC:(NN*nC-1)]') = cmax;
   end
   
   cm = 100 * mean(sum(S0.^2, 2), 1);
   
%    disp(100 - cm);
   
end
%%
cv = S * V;

xs = 1:nC;
xn = linspace(1, nC, nC*10); 

sig = 1;
d0 = (xs' - xs).^2;
d1 = (xn' - xs).^2;

K0 = exp(-d0/sig);
K1 = exp(-d1/sig);

Km = K1 / (K0 + 0.001 * eye(nC));

[cmaxup, iclust] = max(cv * Km', [], 2);

100 * mean(cmaxup.^2)


[~, isort] = sort(iclust);
% 
% % Sm = Ff./std(Ff,1,2)/size(Ff,2).^.5;
% Sm = S;
% 
% Sm = gpuArray(Sm(isort, :));
% 
% Sm = my_conv2(Sm, [20], [1]);
% 
% clf
% imagesc(Sm, [-.2 .7]/150)
% colormap('gray')
%%

