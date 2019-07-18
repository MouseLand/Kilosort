function benchmark_drift_simulation_jrc(GTfilepath, simRecfilepath)

%load in the files (might make this callable later...)
% resPath = 'D:\drift_simulations\74U_norm_64site_no_drift\jrc_qq5_bp_r01\sim_binary.imec.ap_res.mat';
% GTfilepath = 'D:\drift_simulations\74U_norm_64site_no_drift\eMouseGroundTruth.mat';
% simRecfilepath = 'D:\drift_simulations\74U_norm_64site_no_drift\eMouseSimRecord.mat';
% 
% jrc_res = load(resPath);
load(GTfilepath);           %loads in gtClu, gtRes

%Read in spike times from jrc output res.spikeTimes
%Read in spike cluster labels from jrc output res.hClust.spikeClusters

testClu = jrc_res.hClust.spikeClusters;
testRes = jrc_res.spikeTimes;


%JRClust keeps spikes from deleted clusters, with cluster label = -1
%JRClust also keeps unassigned spikes, with label = 0
%Remove all of those

badSpikes = jrc_res.hClust.spikeClusters < 1; %logical array = 1 for spikes w/ cluster label < 1

testClu(badSpikes) = [];
testRes(badSpikes) = [];

[allScores, allFPrates, allMissRates, allMerges, unDetected, overDetected, gtCluIDs] = ...
    compareClustering2_drift(gtClu, gtRes, testClu, testRes, []);

for i = 1:numel(unDetected)
    fprintf('%.3f\n',unDetected(i));
end
    
clid = unique(gtClu);
clear gtimes
for k = 1:length(clid)
    gtimes{k} = double(gtRes(gtClu==clid(k)));
end

%%
autoMiss = (cellfun(@(x) x(1), allMissRates));
autoFP = (cellfun(@(x) x(1), allFPrates));
bestMiss = (cellfun(@(x) x(end), allMissRates));
bestFP = (cellfun(@(x) x(end), allFPrates));

autoScore = ones(1,numel(bestMiss));
autoScore = autoScore - autoMiss -autoFP;
bestScore = ones(1,numel(bestMiss));
bestScore = bestScore - bestMiss - bestFP;
figure

plot(sort(cellfun(@(x) x(1), allFPrates)), '-*b', 'Linewidth', 2)
hold all
plot(sort(cellfun(@(x) x(1), allMissRates)), '-*r', 'Linewidth', 2)
plot(sort(cellfun(@(x) x(end), allFPrates)), 'b', 'Linewidth', 2)
plot(sort(cellfun(@(x) x(end), allMissRates)), 'r', 'Linewidth', 2)
plot(sort(unDetected), 'g', 'Linewidth', 2);
ylim([0 1])
box off

thGood = 0.8;

fprintf('%d / %d good cells, score > %.2f (pre-merge) \n', sum(autoScore > thGood), numel(allScores), thGood)
fprintf('%d / %d good cells, score > %.2f (post-merge) \n', sum(bestScore > thGood), numel(allScores), thGood)

nMerges = cellfun(@(x) numel(x)-1, allMerges);
fprintf('Mean merges per good cell %2.2f \n', mean(nMerges(bestScore > thGood)))
% disp(cellfun(@(x) x(end), allScores))

xlabel('ground truth cluster')
ylabel('fractional error')

legend('false positives (initial)', 'miss rates (initial)', 'false positives (best)', 'miss rates (best)','undetected');
legend boxoff
set(gca, 'Fontsize', 20)
set(gcf, 'Color', 'w')

title('JRC results vs. GT');



hold off;
%%


%sort these in order of the GT label
[sortLabel, sortLabelInd] = sort(gtCluIDs);

bestMiss = bestMiss(sortLabelInd);
bestFP = bestFP(sortLabelInd);

%read in amplitude data for the clusters
%contains yDriftRec nGTSpike x 4 array with
%   spike time in seconds
%   yDrift position
%   GT cluster label
%   nominal amplitude at the monitor site (before mulitplying by factor to
%   create amplitude std)
%   
load(simRecfilepath);
 
%get average amplitude for each cluLabel 
NN = numel(unique(yDriftRec(:,3)));
meanAmp = zeros(1,NN);
meanPos = zeros(1,NN);
nSpike = zeros(1,NN);
%for the special case of simulated data, the labels start at 1 and are
%sequential
for i = 1:NN
    ind = find(yDriftRec(:,3) == i);
    nSpike(i) = numel(ind);
    meanAmp(i) = mean(yDriftRec(ind,4)); 
    meanPos(i) = mean(yDriftRec(ind,2));
end

fprintf('GTlabel\tnSpike\tmeanAmp\tmeanPos\tundetected\tbestMiss\tbestFP\tbestScore\tautoMiss\tautoFP\tautoScore\tnMerges\tphy labels\n');
nMerges = zeros(1,NN);
for i = 1:NN
    nMerges(i) = length(allMerges{i});
    fprintf('%d\t%d\t%.3f\t%.3f\t%.3f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t', ...
    i, nSpike(i), meanAmp(i), ...
    meanPos(i), unDetected(i), bestMiss(i), bestFP(i),bestScore(i), autoMiss(i),...
    autoFP(i), autoScore(i), nMerges(i));
    for j = 1:length(allMerges{i})
        fprintf( '%d,', allMerges{i}(j) );
    end
    fprintf('\n');
        
end

%sort meanAmp to plot missed/FP in order of amplitude
%[sortAmp, sortInd] = sort(meanAmp);
%miss_fp = (vertcat( bestMiss, bestFP ))'
%sort bestMiss and bestFP
%miss_fp = (vertcat( bestMiss(sortInd), bestFP(sortInd), meanAmp(sortInd)
%))';


end