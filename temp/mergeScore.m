function out = mergeScore(x)


s1 = var(x(x>mean(x)));
s2 = var(x(x<mean(x)));

mu1 = mean(x(x>mean(x)));
mu2 = mean(x(x<mean(x)));
p  = mean(x>mean(x));

logp = zeros(numel(x), 2);

rs = zeros(numel(x), 2);
rs(x<0, 1) = 1;
rs(x>0, 2) = 1;

for k = 1:20
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
        
end

ilow = rs(:,1)>rs(:,2);

plow = mean(rs(ilow,1));
phigh = mean(rs(~ilow,2));

% when do I split
out =  ~(plow>.9 && phigh>.9);

% if sign(mu1*mu2)>0
%     out = 0;
% end