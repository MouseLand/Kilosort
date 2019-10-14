function [K, Qi, Q00, Q01, Ri] = ccg(st1, st2, nbins, tbin)
% this function efficiently computes the crosscorrelogram between two sets
% of spikes (st1, st2), with tbin length each, timelags =  plus/minus nbins
% and then estimates how refractory the cross-correlogram is, which can be used
% during merge decisions.


st1 = sort(st1(:)); % makes sure the spike trains are sorted in increasing order
st2 = sort(st2(:));

dt = nbins*tbin; % maximum duration in the cross-correlogram


T = (max([st1; st2])-min([st1; st2])); % total time range

% we traverse both spike trains together, keeping track of the spikes in the first
% spike train that are within dt of spikes in the second spike train
ilow = 1; % lower bound index
ihigh = 1; % higher bound index
j = 1; % index of the considered spike

K = zeros(2*nbins+1, 1);

while j<=numel(st2) % traverse all spikes in the second spike train
    while (ihigh<=numel(st1)) && (st1(ihigh) < st2(j)+dt)
        ihigh = ihigh + 1; % keep increasing higher bound until it's OUTSIDE of dt range
    end
    while (ilow<=numel(st1)) && st1(ilow) <= st2(j)-dt
        ilow = ilow + 1; % keep increasing lower bound until it's INSIDE of dt range
    end
    if ilow>numel(st1)
        break; % break if we exhausted the spikes from the first spike train
    end
    if st1(ilow) > st2(j)+dt
      % if the lower bound is actually outside of dt range, means we overshot (there were no spikes in range)
      % simply move on to next spike from second spike train
        j = j+1;
        continue;
    end
    for k = ilow:(ihigh-1)
      % for all spikes within plus/minus dt range
        ibin = round((st2(j) - st1(k))/tbin); % convert ISI to integer
%         disp(ibin)
        K(ibin+ nbins+1) = K(ibin + nbins+1) + 1; % add 1 to the corresponding correlogram bin
    end
    j = j+1;
end

irange1 = [2:nbins/2 (3/2*nbins+1):(2*nbins)]; % this index range corresponds to the CCG shoulders
irange2 = [nbins+1-50:nbins-10]; % these indices are the narrow, immediate shoulders
irange3 = [nbins+12:nbins+50];

% normalize the shoulders by what's expected from the mean firing rates
% a non-refractive poisson process should yield 1
Q00 = sum(K(irange1)) / (numel(irange1)*tbin* numel(st1) * numel(st2)/T);
Q01 = sum(K(irange2)) / (numel(irange2)*tbin* numel(st1) * numel(st2)/T); % do the same for irange 2
Q01 = max(Q01, sum(K(irange3)) / (numel(irange3)*tbin* numel(st1) * numel(st2)/T)); % compare to the other shoulder

R00 = max(mean(K(irange2)), mean(K(irange3))); % take the biggest shoulder
R00 = max(R00, mean(K(irange1))); % compare this to the asymptotic shoulder

% test the probability that a central area in the autocorrelogram might be refractory
% test increasingly larger areas of the central CCG
a = K(nbins+1);
K(nbins+1) = 0; % overwrite the center of the correlogram with 0 (removes double counted spikes)
for i = 1:10
    irange = [nbins+1-i:nbins+1+i]; % for this central range of the CCG
    Qi0 = sum(K(irange)) / (2*i*tbin* numel(st1) * numel(st2)/T); % compute the same normalized ratio as above. this should be 1 if there is no refractoriness
    Qi(i) = Qi0; % save the normalized probability

    n = sum(K(irange))/2;
    lam = R00 * i;

%     logp = log(lam) * n - lam - gammaln(n+1);

    % this is tricky: we approximate the Poisson likelihood with a gaussian of equal mean and variance
    % that allows us to integrate the probability that we would see <N spikes in the center of the
    % cross-correlogram from a distribution with mean R00*i spikes
    p = 1/2 * (1+ erf((n - lam)/sqrt(2*lam)));

    Ri(i) = p; % keep track of p for each bin size i
end

K(nbins+1) = a; % restore the center value of the cross-correlogram
Qin = Qi/Q00; % normalize the normalized refractory index in two different ways
Qin1 = Qi/Q01;
