function [uNew, nSnew]= reduce_clusters(uS, crit)


cdot = uS(:,:,1)' * uS(:,:,1);
for j = 2:size(uS,3)
    cdot = cdot + uS(:,:,j)' * uS(:,:,j);
end

% compute norms of each spike
newNorms  = sum(sum(uS.^2, 3),1);

% compute sum of pairs of norms
cNorms = 1e-10 + repmat(newNorms', 1, numel(newNorms)) +...
    repmat(newNorms, numel(newNorms), 1);

% compute normalized distance between spikes
cdot = 1 - 2*cdot./cNorms;
cdot = cdot + diag(Inf * diag(cdot));

[cmin, newind] = min(single(cdot>crit),[],1);
% if someone else votes you in, your votee doesn't count
% newind(ismember(1:nN, newind)) = [];
newind = unique(newind(cmin<.5));
if ~isempty(newind)
    newind = cat(2, newind, find(cmin>.5));
else
    newind = find(cmin>.5);
end


uNew = uS(:,newind, :);

nNew = size(uNew,2);

nSnew = merge_spikes_in(uNew, zeros(nNew, 1), uS, crit);