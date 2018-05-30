function [dWU, nnew] = triageKmeans(dWU, nsp, uproj, ioff, dxdnu)

Nfilt = size(dWU,2);

idrop = nsp<5;

% drop low firing neurons
dWU(:,idrop,:) = [];
nsp(idrop) = [];

cc = dWU' *  dWU;
cdiag = diag(cc);
cc = cc ./(cdiag(:) * cdiag(:)');
cc = cc - diag(diag(cc));

rdir = (nsp(:) - nsp(:)')<0;

ipair = (cc>0.9 & rdir);
amax = max(ipair, [], 2);

idrop = amax>0;

dWU(:,idrop,:) = [];
nsp(idrop) = [];

Th = linspace(.25, .5, 5);
k = 0;
nFeat = size(uproj,1);
nnew = Nfilt - size(dWU,2);

while 1
    k = k+1;
    
    N0 = size(dWU,2);
    if N0>=Nfilt
       break; 
    end
    
    inew = find(dxdnu < Th(k), Nfilt-N0);

    for j = numel(inew):-1:1
        dWU(double(ioff(inew(j))) + [1:nFeat], j + N0) = uproj(:, inew(j));        
    end
       
end
