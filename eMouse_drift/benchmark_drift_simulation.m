function benchmark_drift_simulation(rez, GTfilepath, simRecfilepath)

load(GTfilepath)


testClu = rez.st3(:,2) ;
flag = 0;
testRes = rez.st3(:,1);

[testRes, tOrder] = sort(testRes);
testClu = testClu(tOrder);

[allScores, allFPrates, allMissRates, allMerges, gtCluIDs] = ...
    compareClustering2_drift(gtClu, gtRes, testClu, testRes, []);

%


clid = unique(gtClu);
clear gtimes
for k = 1:length(clid)
    gtimes{k} = double(gtRes(gtClu==clid(k)));
end

%% figure and output results

figure

plot(sort(cellfun(@(x) x(1), allFPrates)), '-*b', 'Linewidth', 2)
hold all
plot(sort(cellfun(@(x) x(1), allMissRates)), '-*r', 'Linewidth', 2)
plot(sort(cellfun(@(x) x(end), allFPrates)), 'b', 'Linewidth', 2)
plot(sort(cellfun(@(x) x(end), allMissRates)), 'r', 'Linewidth', 2)
ylim([0 1])
box off

finalScores = cellfun(@(x) x(end), allScores);
fprintf('%d / %d good cells, score > 0.8 (pre-merge) \n', sum(cellfun(@(x) x(1), allScores)>.8), numel(allScores))
fprintf('%d / %d good cells, score > 0.8 (post-merge) \n', sum(cellfun(@(x) x(end), allScores)>.8), numel(allScores))

nMerges = cellfun(@(x) numel(x)-1, allMerges);
fprintf('Mean merges per good cell %2.2f \n', mean(nMerges(finalScores>.8)))

% disp(cellfun(@(x) x(end), allScores))

xlabel('ground truth cluster')
ylabel('fractional error')

legend('false positives (initial)', 'miss rates (initial)', 'false positives (best)', 'miss rates (best)')
legend boxoff
set(gca, 'Fontsize', 20)
set(gcf, 'Color', 'w')

title('Kilosort2 Results') 

hold off;

%Calculate results vs. known unit properties

bestMiss = (cellfun(@(x) x(end), allMissRates));
bestFP = (cellfun(@(x) x(end), allFPrates));

%sort these in order of the GT label
[~, sortLabelInd] = sort(gtCluIDs);

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
%for the special case of simulated data, the labels start at 1 and are
%sequential
for i = 1:NN
    ind = find(yDriftRec(:,3) == i);
    meanAmp(i) = mean(yDriftRec(ind,4));
    meanPos(i) = mean(yDriftRec(ind,2));
end

fprintf('GTlabel\tmeanAmp\tmeanPos\tbestMiss\tbestFP\tnMerges\tphy labels\n')
nMerges = zeros(1,NN);
for i = 1:NN
    nMerges(i) = length(allMerges{i})-1;
    fprintf('%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t', i, meanAmp(i), meanPos(i), bestMiss(i), bestFP(i),nMerges(i));
    for j = 1:length(allMerges{i})-1
        fprintf( '%d,', allMerges{i}(j)-1 );
    end
    fprintf('%d\n',allMerges{i}(length(allMerges{i}))-1);       
end


end