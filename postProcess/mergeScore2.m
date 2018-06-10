function mh = mergeScore2(x)


x1 = -100;
x2 = 100;


bins = linspace(x1, x2, 1001);

his = histc(x, bins);

imax = ceil(numel(his)/2);

[mh, imax] = split_histograms(his, imax);