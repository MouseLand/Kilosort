function [nS, iNonMatch] = merge_spikes0(uBase, nS, uS, crit)

if ~isempty(uBase)
    cdot = uBase * uS';
   
    baseNorms = sum(uBase.^2, 2)';
    newNorms  = sum(uS.^2, 2)';
    
    cNorms = 1e-10 + repmat(baseNorms', 1, numel(newNorms)) + repmat(newNorms, numel(baseNorms), 1);
    
    cdot = 1 - 2*cdot./cNorms;
    
    [cdotmin, imin] = min(cdot, [], 1);
    
    iMatch = cdotmin<crit;
    
    nSnew = hist(imin(iMatch), 1:1:size(uBase,1));
    nS = nS + nSnew';
    
    
    iNonMatch = find(cdotmin>crit);
else
   iNonMatch = 1:size(uS,2); 
   nS = [];
end