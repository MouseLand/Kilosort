function [iclustup, isort] = embed1D(S, nC, isort)

xs = 1:nC;
xn = linspace(1, nC, nC*100);

sig = 1;
d0 = (xs' - xs).^2;
d1 = (xn' - xs).^2;

K0 = exp(-d0/sig);
K1 = exp(-d1/sig);

Km = K1 / (K0 + 0.001 * eye(nC));
[NN, NT] = size(S);


iclust = zeros(NN, 1);

nn = floor(NN/nC);
iclust(isort) = ceil([1:NN]/nn);
iclust(iclust>nC) = nC;

sig = [linspace(3,1,25) 1*ones(1,50)];

for t = 1:numel(sig)
    V = gpuArray.zeros(NT, nC, 'single');
    for j = 1:nC
        ix = iclust== j;
        V(:, j) = sum(S(ix, :),1);
    end
    
    V = my_conv2(V, sig(t), 2);
    V = normc(V);
    
    cv = S * V;
    [cmax, iclust] = max(cv, [], 2);
    
    %disp(mean(100 * cmax.^2));
    
end

[cmaxup, iclustup] = max(cv * Km', [], 2);
iclustup = gather(iclustup);
[~, isort] = sort(iclustup);