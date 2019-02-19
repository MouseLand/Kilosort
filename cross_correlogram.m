function acg = cross_correlogram(st1, st2, nbins, tbin)


dt = nbins*tbin;

% if numel(st1)<numel(st2)
%     
% end

ilow = 1;
ihigh = 1;
j = 1;

acg = zeros(2*nbins+1, 1); 

while j<=numel(st2)
    while st1(ihigh) < st2(j)+dt
        ihigh = ihigh + 1;
    end
    while st1(ilow) <= st2(j)-dt
        ilow = ilow + 1;
    end
    for k = [ilow:j-1 j+1:ihigh-1]
        ibin = round((st2(j) - st1(k))/tbin);
        acg(ibin+ nbins+1) = K(ibin + nbins+1) + 1;
    end
    j = j+1;
end
    
