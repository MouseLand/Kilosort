addpath('D:\DATA\Spikes\EvaluationCode')
% datFilename = '20150601_all.dat';
%%
dd = load('C:\DATA\Spikes\set7\20151102_1_ks_results.mat');
root        = fullfile('C:\DATA\Spikes', sprintf('set%d', idset));

fname       = 'spike_clustersSorted.npy';
    
gtCluMan = readNPY(fullfile(root, fname)); 

totM = zeros(1000, 1);
for i = 1:1000
   totM(i) = numel(unique(dd.rez.st3(gtCluMan==i, 2))); 
   nsp(i) = sum(gtCluMan==i);
end
%%
igt = ismember(gtCluMan, find(totM==1 & nsp>1000));

% igt = ismember(gtCluMan, 563);

gtRes = double(dd.rez.st3(igt, 1));
gtClu = double(gtCluMan(igt, 1));
%%
testRes = st3(:,1); 
testClu = st3(:,2);

[allScores, allFPrates, allMissRates, allMerges] = ...
    compareClustering(gtClu, gtRes, testClu, testRes, []);
%
clid = unique(gtClu);
clear gtimes
for k = 1:length(clid)
    gtimes{k} = double(gtRes(gtClu==clid(k)));
end


%%
bestPostMerge = [];
for j = 1:length(allScores)
    bestPostMerge(j) = allScores{j}(end);
end