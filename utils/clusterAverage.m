

function clusterQuantity = clusterAverage(clu, spikeQuantity)
% function clusterQuantity = clusterAverage(clu, spikeQuantity)
%
% get the average of some quantity across spikes in each cluster, given the
% quantity for each spike
%
% e.g. 
% > clusterDepths = clusterAverage(clu, spikeDepths);
%
% clu and spikeQuantity must be vector, same size

% using a super-tricky algorithm for this - when you make a sparse
% array, the values of any duplicate indices are added. So this is the
% fastest way I know to make the sum of the entries of spikeQuantity for each of
% the unique entries of clu
[~, spikeCounts] = countUnique(clu);

% convert clu to indices, i.e. just values between 1 and nClusters. 
[~,~,cluInds] = unique(clu);

% summation
q = full(sparse(cluInds, ones(size(clu)), double(spikeQuantity))); 

% had sums, so dividing by spike counts gives the mean depth of each cluster
clusterQuantity = q./spikeCounts; 
