function [nS, iNonMatch] = merge_spikes_in(uBase, nS, uS, crit)

if ~isempty(uBase)
    cdot = uBase(:,:,1)' * uS(:,:,1);
    for j = 2:size(uBase,3)
        cdot = cdot + uBase(:,:,j)' * uS(:,:,j);
    end
    
    baseNorms = sum(sum(uBase.^2, 3),1);
    newNorms  = sum(sum(uS.^2, 3),1);
    
    cNorms = 1e-10 + repmat(baseNorms', 1, numel(newNorms)) + repmat(newNorms, numel(baseNorms), 1);
    
    cdot = 1 - 2*cdot./cNorms;
    
    [cdotmin, imin] = min(cdot, [], 1);
    
    iMatch = cdotmin<crit;
    
    nSnew = hist(imin(iMatch), 1:1:size(uBase,2));
    nS = nS + nSnew';
    % for j = 1:size(uBase,2)
    %    inds = find(iMatch & imin==j);
    %    nNew = numel(inds);
    %    uBase(:,j,:) = uBase(:,j,:) * nS(j)/(nS(j) + nNew) + ...
    %        sum(uS(:,inds,:),2)/(nS(j) + nNew);
    %    nS(j) = nS(j) + nNew;
    % end
    
    
    iNonMatch = find(cdotmin>crit);
else
   iNonMatch = 1:size(uS,2); 
   nS = [];
end