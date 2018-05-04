function steps = merging_score(fold, fnew, fracse)


troughToPeakRatio = 3;

l1 = min(fnew);
l2 = max(fold);

se = (std(fold) + std(fnew))/2;
se25 = fracse * se;
b2 = [0:se25:-l1];
b1 = [0:se25:l2];

hs1 = my_conv(histc(fold, b1), 1);
hs2 = my_conv(histc(-fnew, b2), 1);

mmax = min(max(hs1), max(hs2));

m1 = ceil(mean(fold)/se25);
m2 = -ceil(mean(fnew)/se25);

steps = sum(hs1(1:m1)<mmax/troughToPeakRatio) + ...
    sum(hs2(1:m2)<mmax/troughToPeakRatio);

