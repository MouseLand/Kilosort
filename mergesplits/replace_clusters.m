function [dWUtot, dbins, nswitch, nspikes, iYout] = ...
    replace_clusters(dWUtot,dbins, Nbatch, mergeT, splitT,  WUinit, nspikes)

uu = Nbatch * dbins;
nhist = 1:1:100;
% nSpikes = sum(uu,1);
nSpikes = sum(nspikes,2)';

[score, iY1, mu1, mu2, u1, u2]   = split_clust(uu, nhist);
[~, iY, drez]                    = distance_betwxt(dWUtot);

[dsort, isort] = sort(drez, 'ascend');
iYout = iY(isort);

nmerged = sum(dsort<mergeT);
nsplit = sum(score>splitT);

mu = sum(sum(dWUtot.^2,1),2).^.5;
mu = mu(:);
freeInd = find(nSpikes<200 | mu'<10 | isnan(mu'));

for k = 1:nmerged
    % merge the two clusters
    iMerged = iY(isort(k));
    wt = [nSpikes(iMerged); nSpikes(isort(k))];
    wt = wt/sum(wt);
%     mu(iMerged) = [mu(iMerged) mu(isort(k))] * wt;
    
    dWUtot(:,:,iMerged)  = dWUtot(:,:,iMerged) * wt(1) + dWUtot(:,:,isort(k)) * wt(2);
    dWUtot(:,:,isort(k)) = 1e-10;
    
    nspikes(iMerged, :) = nspikes(iMerged, :) + nspikes(isort(k), :);
    nspikes(isort(k), :) = 0;
end


for k = 1:min(nmerged+numel(freeInd), nsplit)
    if k<=numel(freeInd)
        inew= freeInd(k);
    else
        inew = isort(k - numel(freeInd));
    end
    
    mu0 = mu(iY1(k));
    
    % split the bimodal cluster, overwrite merged cluster
    mu(inew)     = mu1(k);
    mu(iY1(k))   = mu2(k);
    
    dbins(:, inew)     = u1(:, k) /Nbatch;
    dbins(:, iY1(k))   = u2(:, k) /Nbatch;

    nspikes(inew, :)     = nspikes(iY1(k), :)/2;
    nspikes(iY1(k), :)   = nspikes(iY1(k), :)/2;
    dWUtot(:,:,inew)     = mu1(k)/mu0 * dWUtot(:,:,iY1(k)); %/npm(iY1(k));
    dWUtot(:,:,iY1(k))   = mu2(k)/mu0 * dWUtot(:,:,iY1(k)); %/npm(iY1(k));
end

d2d                 = pairwise_dists(dWUtot, WUinit);
dmatch              = min(d2d, [], 1);

[~, inovel] = sort(dmatch, 'descend');
% inovel = find(dmatch(1:1000)>.4);
% inovel = inovel(randperm(numel(inovel)));

i0 = 0;

for k = 1+min(nmerged+numel(freeInd), nsplit):nmerged+numel(freeInd)
    % add new clusters
    i0 = i0 + 1;
    if i0>numel(inovel)
        break;
    end
    if k<=numel(freeInd)
        inew= freeInd(k);
    else
        inew = isort(k - numel(freeInd));
    end
     
    dbins(:, inew)     = 1;
    
    nspikes(inew, :) = 1/8;
    
    
    dWUtot(:,:,inew)     = WUinit(:,:,inovel(i0)); %ratio * mu1(k)/mu0 * dWUtot(:,:,iY1(k));
    
end

nswitch = [min(nmerged, nsplit) i0]; %min(nmerged+numel(freeInd), nsplit);

