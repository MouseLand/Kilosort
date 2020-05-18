
k = ceil(rand*Nfilt);

ix = rez.st3(:,2)==(k+1);

xs = rez.st3(ix, 4);

binx = exp(linspace(log(.1), log(10), 100));

clf
ds = histc(xs, binx);
semilogx(binx, ds)