function [K, Qi, Q00, Q01, Ri] = ccg(st1, st2, nbins, tbin)

st1 = sort(st1(:));
st2 = sort(st2(:));

dt = nbins*tbin;

% if numel(st1)<numel(st2)
%     
% end

T = (max([st1; st2])-min([st1; st2]));

ilow = 1;
ihigh = 1;
j = 1;

K = zeros(2*nbins+1, 1); 

while j<=numel(st2)
%     disp(j)
    while (ihigh<=numel(st1)) && (st1(ihigh) < st2(j)+dt)
        ihigh = ihigh + 1;
    end
    while (ilow<=numel(st1)) && st1(ilow) <= st2(j)-dt
        ilow = ilow + 1;
    end
    if ilow>numel(st1)
        break;
    end
    if st1(ilow) > st2(j)+dt
        j = j+1;
        continue;
    end
    for k = ilow:(ihigh-1)
        ibin = round((st2(j) - st1(k))/tbin);
%         disp(ibin)
        K(ibin+ nbins+1) = K(ibin + nbins+1) + 1;
    end
    j = j+1;
end

irange1 = [2:nbins/2 (3/2*nbins+1):(2*nbins)];
Q00 = sum(K(irange1)) / (numel(irange1)*tbin* numel(st1) * numel(st2)/T);

irange2 = [nbins+1-50:nbins-10];
irange3 = [nbins+12:nbins+50];

R00 = max(mean(K(irange2)), mean(K(irange3)));
R00 = max(R00, mean(K(irange1)));

Q01 = sum(K(irange2)) / (numel(irange2)*tbin* numel(st1) * numel(st2)/T);
Q01 = max(Q01, sum(K(irange3)) / (numel(irange3)*tbin* numel(st1) * numel(st2)/T));

% disp(R00)

K(nbins+1) = 0;
for i = 1:10
    irange = [nbins+1-i:nbins+1+i];
    Qi0 = sum(K(irange)) / (2*i*tbin* numel(st1) * numel(st2)/T);
    Qi(i) = Qi0;
    
    n = sum(K(irange))/2;
    lam = R00 * i;
    
%     logp = log(lam) * n - lam - gammaln(n+1);
    p = 1/2 * (1+ erf((n - lam)/sqrt(2*lam)));
    
    Ri(i) = p;
    
end


Qin = Qi/Q00;
Qin1 = Qi/Q01;