function rez = set_cutoff(rez)
% after everything else is done, this function takes spike trains and cuts off
% any noise they might have picked up at low amplitude values
% We look for bimodality in the amplitude plot, thus setting an individual threshold
% for each neuron.
% Also, this function calls "good" and "bad" clusters based on the auto-correlogram

ops = rez.ops;
dt = 1/1000; % step size for CCG binning

Nk = numel(rez.mu); % number of templates

% sort by firing rate first
rez.good = zeros(Nk, 1);
rez.est_contam_rate = ones(Nk, 1);

for j = 1:Nk
    ix = find(rez.st3(:,2)==j); % find all spikes from this neuron
    ss = rez.st3(ix,1)/ops.fs; % convert to seconds
    if numel(ss)==0
        continue; % break if there are no spikes
    end

    vexp = rez.st3(ix,4); % vexp is the relative residual variance of the spikes

    Th = ops.Th(1); % start with a high threshold

    fcontamination = 0.1; % acceptable contamination rate

    rez.est_contam_rate(j) = 1;
    while Th>=ops.Th(2)
      % continually lower the threshold, while the estimated unit contamination is low
        st = ss(vexp>Th); % take spikes above the current threshold
        if isempty(st)
            Th = Th - .5; % if there are no spikes, we need to keep lowering the threshold
            continue;
        end
        [K, Qi, Q00, Q01, rir] = ccg(st, st, 500, dt); % % compute the auto-correlogram with 500 bins at 1ms bins
        Q = min(Qi/(max(Q00, Q01))); % this is a measure of refractoriness
        R = min(rir); % this is a second measure of refractoriness (kicks in for very low firing rates)
        if Q>fcontamination || R>.05 % if the unit is already contaminated, we break, and use the next higher threshold
           break;
        else
            if Th==ops.Th(1) && Q<.05
              % only on the first iteration, we consider if the unit starts well isolated
              % if it does, then we put much stricter criteria for isolation
              % to make sure we don't settle for a relatively high contamination unit
                fcontamination = min(.05, max(.01, Q*2));

                % if the unit starts out contaminated, we will settle with the higher contamination rate
            end
            rez.good(j) = 1; % this unit is good, because we will stop lowering the threshold when it becomes bad
            Th = Th - .5; % try the next lower threshold
        end
    end
    Th = Th + .5; % we exited the loop because the contamination was too high. We revert to the higher threshold
    st = ss(vexp>Th); % take spikes above the current threshold
    [K, Qi, Q00, Q01, rir] = ccg(st, st, 500, dt); % % compute the auto-correlogram with 500 bins at 1ms bins
    Q = min(Qi/(max(Q00, Q01))); % this is a measure of refractoriness
    rez.est_contam_rate(j) = Q; % this score will be displayed in Phy

    rez.Ths(j) = Th; % store the threshold for potential debugging
    rez.st3(ix(vexp<=Th), 2) = 0; % any spikes below the threshold get discarded into a 0-th cluster.
end

% we sometimes get NaNs, why? replace with full contamination
rez.est_contam_rate(isnan(rez.est_contam_rate)) = 1;

% remove spikes from the 0th cluster
rez = remove_spikes(rez,rez.st3(:,2)==0,'below_cutoff');
