function [uNew, nSnew]= reduce_clusters0(uS, crit)

 cdot = uS * uS';
 
% compute norms of each spike
newNorms  = sum(uS.^2, 2)';

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


uNew = uS(newind, :);

nNew = size(uNew,1);

nSnew = merge_spikes0(uNew, zeros(nNew, 1), uS, crit);