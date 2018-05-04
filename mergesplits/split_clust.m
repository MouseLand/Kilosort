function [score, iY, mu1, mu2, u1, u2] = split_clust(uu, nhist)

nhist = nhist(:);

nspikes = sum(uu, 1);

uc = zeros(size(uu));
for i = 1:size(uu,2)
    uc(:,i) = my_conv(uu(:,i)',  max(.5, min(4, 2000/nspikes(i))))'; %.5
%       uc(:,i) = my_conv2(uu(:,i),  max(.25, min(4, 2000/nspikes(i))), 1);
end
%
uc = uc ./repmat(sum(uc,1),size(uc,1), 1);
ucum = cumsum(uc, 1);
%
dd = diff(uc, 1);

iY = zeros(1000,1);
mu1 = zeros(1000,1);
mu2 = zeros(1000,1);
var1 = zeros(1000,1);
var2 = zeros(1000,1);
u1 = zeros(size(uu,1), 1000);
u2 = zeros(size(uu,1), 1000);

maxM = max(uc, [], 1);

inew = 0;

Nfilt = size(uu,2);
mu0 = sum(repmat(nhist(1:100, 1), 1, Nfilt) .* uc, 1);
var0 = sum((repmat(nhist(1:100), 1, Nfilt) - repmat(mu0, 100, 1)).^2 .* uc, 1);

for i = 1:Nfilt
    ix = find(dd(1:end-1, i)<0 & dd(2:end, i)>0);
    
    ix = ix(ucum(ix, i)>.1 & ucum(ix, i)<.8 & uc(ix,i)<.8 * maxM(i)); %.9 not .95
    if nspikes(i) > 500 && numel(ix)>0
        ix = ix(1);
        
        inew = inew + 1;
        
        normuc    = sum(uc(1:ix, i));
        mu1(inew) = sum(nhist(1:ix)     .* uc(1:ix, i))    /normuc;
        mu2(inew) = sum(nhist(1+ix:100) .* uc(1+ix:100, i))/(1-normuc);
        
        var1(inew) = sum((nhist(1:ix)-mu1(inew)).^2     .* uc(1:ix, i))    /normuc;
        var2(inew) = sum((nhist(1+ix:100)-mu2(inew)).^2 .* uc(1+ix:100, i))/(1-normuc);
        
        u1(1:ix,inew) = uu(1:ix, i);
        u2(1+ix:100,inew) = uu(1+ix:100, i);
        
        iY(inew) = i;
    end
    
end

mu1 = mu1(1:inew);
mu2 = mu2(1:inew);
var1 = var1(1:inew);
var2 = var2(1:inew);
u1 = u1(:,1:inew);
u2 = u2(:,1:inew);

n1 = sum(u1,1)';
n2 = sum(u2,1)';
iY = iY(1:inew);

score = 1 - (n1.*var1 + n2.*var2)./((n1+n2).*var0(iY)');
% score = ((n1+n2).*var0(iY)' - (n1.*var1 + n2.*var2))./var0(iY)';
[~, isort] = sort(score, 'descend');

iY = iY(isort);
mu1 = mu1(isort);
mu2 = mu2(isort);
u1 = u1(:,isort);
u2 = u2(:,isort);
score = score(isort);
