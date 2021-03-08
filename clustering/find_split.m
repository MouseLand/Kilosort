function [r, scmax, p, m0, mu1, mu2, sig] = find_split(x)

x = gather(x);
qbar = [.001, .999];
nbins = 1001;

xq = quantile(x, qbar);
bins = linspace(single(xq(1)), single(xq(2)), nbins+1);

% bins = (bins(2:end) + (bins(1:end-1))/2;

xhist = histcounts(x, bins);
xhist = single(xhist);
xhist = my_conv2(xhist, 10, 2);

nt = length(xhist);
ival = ones(1, nt, 'single');
ival([1, nt]) = 0;
ival(1:nt-1) = ival(1:nt-1) .* (xhist(1:nt-1) < xhist(2:nt));
ival(2:nt)   = ival(2:nt)   .* (xhist(2:nt) < xhist(1:nt-1));

imin = find(ival);
xmean = mean(x);

if isempty(imin)
    [~, imax] = min(abs(bins(1:nbins-1)-xmean));
    scmax = 0;
    isplit = imax;
    m0 = xmean;
else
    sc = zeros(length(imin),  1, 'single');
    for j =1:length(imin)
        d1 = max(xhist(1:imin(j))) - xhist(imin(j));
        d2 = max(xhist(1+imin(j):nt)) - xhist(imin(j));
        sc(j) = max(0, d1*d2)^.5;
    end
    [scmax, imax] = max(sc);
    m0 = bins(imin(imax));
    isplit = imin(imax);
end

[~, i1] = max(xhist(1:isplit));
[~, i2] = max(xhist(1+isplit:nt));
i2 = isplit + i2;
r = 1 - xhist(isplit)./xhist([i1, i2]);

if nargout>2
    ix = x < bins(isplit);
    p = sum(ix) / length(x);
    mu1 = mean(x(ix));
    mu2 = mean(x(~ix));
    sig = p * var(x(ix)) + (1-p) * var(x(~ix));
    sig = max(sig, var(x)/10);
end