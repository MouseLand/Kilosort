function rez = find_merges(rez, flag)
% this function merges clusters based on template correlation
% however, a merge is veto-ed if refractory period violations are introduced

wPCA  = rez.wPCA;
wroll = [];
tlag = [-2, -1, 1, 2];
for j = 1:length(tlag)
    wroll(:,:,j) = circshift(wPCA, tlag(j), 1)' * wPCA;
end


ops = rez.ops;
dt = 1/1000;

dmu = 2 * abs(rez.mu' - rez.mu ) ./ (rez.mu' + rez.mu);


U = permute(rez.U, [2,1,3]);
W = permute(rez.W, [2,1,3]);
simScore = (U(:,:) * U(:,:)') .* (W(:,:) * W(:,:)')/6;

for j = 1:size(wroll,3)
    [Nfilt, nt0, ~] = size(W);
    Wr = reshape(W, [Nfilt * nt0, 6]);
    Wr = Wr * wroll(:,:,j)';
    Wr = reshape(Wr, [Nfilt, nt0, 6]);
    Xsim =  (U(:,:) * U(:,:)') .* (Wr(:,:) * W(:,:)')/6;
    simScore = max(simScore, Xsim); 
end
rez.simScore = simScore;

Xsim = simScore; % .* (dmu < .2);

Nk = size(Xsim,1);
Xsim = Xsim - diag(diag(Xsim)); % remove the diagonal of ones


% sort by firing rate first
nspk = accumarray(rez.st3(:,2), 1, [Nk, 1], @sum);
[~, isort] = sort(nspk); % we traverse the set of neurons in ascending order of firing rates

fprintf('initialized spike counts\n')

if ~flag
  % if the flag is off, then no merges are performed
  % this function is then just used to compute cross- and auto- correlograms
   rez.R_CCG = Inf * ones(Nk);
   rez.Q_CCG = Inf * ones(Nk);
   rez.K_CCG = {};
end

for j = 1:Nk
    s1 = rez.st3(rez.st3(:,2)==isort(j), 1)/ops.fs; % find all spikes from this cluster
    if numel(s1)~=nspk(isort(j))
        fprintf('lost track of spike counts') %this is a check for myself to make sure new cluster are combined correctly into bigger clusters
    end
    % sort all the pairs of this neuron, discarding any that have fewer spikes
    [ccsort, ix] = sort(Xsim(isort(j),:) .* (nspk'>numel(s1)), 'descend');
    ienu = find(ccsort<.7, 1) - 1; % find the first pair which has too low of a correlation

    
    % for all pairs above 0.5 correlation
    for k = 1:ienu
        s2 = rez.st3(rez.st3(:,2)==ix(k), 1)/ops.fs; % find the spikes of the pair
        % compute cross-correlograms, refractoriness scores (Qi and rir), and normalization for these scores
        [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt);
        Q = min(Qi/(max(Q00, Q01))); % normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
        R = min(rir); % R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes

        if flag
            if Q<.2 && R<.05 % if both refractory criteria are met
                i = ix(k);
                % now merge j into i and move on
                rez.st3(rez.st3(:,2)==isort(j),2) = i; % simply overwrite all the spikes of neuron j with i (i>j by construction)
                nspk(i) = nspk(i) + nspk(isort(j)); % update number of spikes for cluster i
                fprintf('merged %d into %d \n', isort(j), i)
                % YOU REALLY SHOULD MAKE SURE THE PC CHANNELS MATCH HERE
                break; % if a pair is found, we don't need to keep going (we'll revisit this cluster when we get to the merged cluster)
            end
        else
          % sometimes we just want to get the refractory scores and CCG
            rez.R_CCG(isort(j), ix(k)) = R;
            rez.Q_CCG(isort(j), ix(k)) = Q;

            rez.K_CCG{isort(j), ix(k)} = K;
            rez.K_CCG{ix(k), isort(j)} = K(end:-1:1); % the CCG is "antisymmetrical"
        end
    end
end

if ~flag
    rez.R_CCG  = min(rez.R_CCG , rez.R_CCG'); % symmetrize the scores
    rez.Q_CCG  = min(rez.Q_CCG , rez.Q_CCG');
end

hid = int32(rez.st3(:,2));
ss = double(rez.st3(:,1)) / ops.fs;

clust_good = check_clusters(hid, ss, 0.2);
sum(clust_good)
rez.good = clust_good;


