% function out = splitGaussians(xs)

% ik = 1;
nSpikes = 0;
while nSpikes<300
    ik = ceil(rand * Nfilt);
    
    sp = find(rez.st3(:,2)==ik);
    sp = sort(sp);
    
    nSpikes = numel(sp);
    
    
end

clp = rez.cProjPC(sp, :, :);
clp = clp - mean(clp,1);
clp = gpuArray(clp(:,:));
[u s v] = svdecon(clp');

w = u(:,1);
x = gather(clp * w);

s1 = var(x(x>mean(x)));
s2 = var(x(x<mean(x)));

mu1 = mean(x(x>mean(x)));
mu2 = mean(x(x<mean(x)));
p  = .5;

logp = zeros(numel(sp), 2);


for k = 1:50
    
    logp(:,1) = -1/2*log(s1) - (x-mu1).^2/(2*s1) + log(p);
    logp(:,2) = -1/2*log(s2) - (x-mu2).^2/(2*s2) + log(1-p);
    
    lMax = max(logp,[],2);
    logp = logp - lMax;
    
    rs = exp(logp);
    
    pval = log(sum(rs,2)) + lMax;
    logP(k) = mean(pval);
    
    rs = rs./sum(rs,2);
    
    p = mean(rs(:,1));
    mu1 = (rs(:,1)' * x )/sum(rs(:,1));
    mu2 = (rs(:,2)' * x )/sum(rs(:,2));
    
    s1 = (rs(:,1)' * (x-mu1).^2 )/sum(rs(:,1));
    s2 = (rs(:,2)' * (x-mu2).^2 )/sum(rs(:,2));
    
    if (k>10 && rem(k,2)==1)
        StS  = clp' * (clp .* (rs(:,1)/s1 + rs(:,2)/s2))/nSpikes;
        StMu = clp' * (rs(:,1)*mu1/s1 + rs(:,2)*mu2/s2)/nSpikes;
        
        w = StMu'/StS;
        w = normc(w');
        x = gather(clp * w);
    end
end

figure(1)
subplot(1,4,1)
plot(logP(1:k))

subplot(1,4,2)
[~, isort] = sort(x);
epval = exp(pval);
epval = epval/sum(epval);
plot(x(isort), epval(isort))

subplot(1,4,3)
ts = linspace(min(x), max(x), 200);
xbin = hist(x, ts);
xbin = xbin/sum(xbin);

plot(ts, xbin)

figure(2)
plotmatrix(v(:,1:4), '.')

drawnow

% compute scores for splits
ilow = rs(:,1)>rs(:,2);
ps = mean(rs(:,1));
[mean(rs(ilow,1)) mean(rs(~ilow,2)) max(ps, 1-ps) min(mean(ilow), mean(~ilow))]





