function igood = check_clusters(hid, ss, rmax)

nmax = max(hid);
igood = zeros(nmax,1);
for j = 1:nmax
    ix = hid==j;
    if sum(ix)>0
        igood(j) = is_good(ss(ix), rmax);    
    end
end
end

function igood = is_good(ss, rmax)
dt = 1/1000;
fcontamination = 0.1;
[K, Qi, Q00, Q01, rir] = ccg(ss, ss, 500, dt); % % compute the auto-correlogram with 500 bins at 1ms bins
Q = min(Qi/(max(Q00, Q01))); % this is a measure of refractoriness
R = min(rir); % this is a second measure of refractoriness (kicks in for very low firing rates)
if Q<fcontamination && R<rmax
   igood = 1;  
else
    igood = 0;
end

end