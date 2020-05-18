function benchmark_drift_simulation(rez, GTfilepath, simRecfilepath, sortType, bAutoMerge, varargin)

bOutFile = 0;

%these definitions for testing outside a script. comment out for normal calling!
%can leave out last 2 lines if output file not desired
% load('D:\test_new_sim\74U_20um_drift_standard\r28_KS2determ_r26rep\rez2.mat');
% GTfilepath = 'D:\test_new_sim\74U_20um_drift_standard\eMouseGroundTruth.mat';
% simRecfilepath = 'D:\test_new_sim\74U_20um_drift_standard\eMouseSimRecord.mat';
% bOutFile = 1;
% out_fid = fopen('D:\test_new_sim\74U_20um_drift_standard\r28_benchmark_output.txt','w');

sortType = 2;
bAutoMerge = 0;

load(GTfilepath);

if bAutoMerge
    testClu = 1 + rez.st3(:,5) ;
else
    testClu = rez.st3(:,2) ;
end


%fprintf( 'length of vargin: %d\n', numel(varargin));
if( numel(varargin) == 1)
    %path for output file
    bOutFile = 1;
    fprintf( 'output filename: %s\n', varargin{1} );
    out_fid = fopen( varargin{1}, 'w' );
end
    

testRes = rez.st3(:,1);

[testRes, tOrder] = sort(testRes);
testClu = testClu(tOrder);

%get cluster position footprint for each
[cluPos, unitSize] = getFootPrintRez(rez);

[allScores, allFPrates, allMissRates, allMerges, unDetected, overDetected, gtCluIDs] = ...
    compareClustering2_drift(gtClu, gtRes, testClu, testRes, []);

%


clid = unique(gtClu);
clear gtimes
for k = 1:length(clid)
    gtimes{k} = double(gtRes(gtClu==clid(k)));
end

%% figure and output results
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
%plot(sort(overDetected), '-g', 'Linewidth', 2);
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

legend('false positives (initial)', 'miss rates (initial)', 'false positives (best)', 'miss rates (best)', 'undetected')
legend boxoff
set(gca, 'Fontsize', 15)
set(gcf, 'Color', 'w')

if bAutoMerge
    titleStr = sprintf('Kilosort%d with Automerge', sortType);
else
    titleStr = sprintf('Kilosort%d Results', sortType);
end
title(titleStr)

hold off;

%Calculate results vs. known unit properties



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
nSpike = zeros(1,NN);
%for the special case of simulated data, the labels start at 1 and are
%sequential
for i = 1:NN
    ind = find(yDriftRec(:,3) == i);
    nSpike(i) = numel(ind);
    meanAmp(i) = mean(yDriftRec(ind,4));
    meanPos(i) = mean(yDriftRec(ind,2));
end

if( bOutFile )
    fprintf(out_fid, 'GTlabel\tnSpike\tmeanAmp\tmeanPos\tundetected\tbestMiss\tbestFP\tbestScore\tautoMiss\tautoFP\tautoScore\tnMerges\tphy labels\n');
else
    fprintf('GTlabel\tnSpike\tmeanAmp\tmeanPos\tundetected\tbestMiss\tbestFP\tbestScore\tautoMiss\tautoFP\tautoScore\tnMerges\tphy labels\n');
end

nMerges = zeros(1,NN);
for i = 1:NN
    nMerges(i) = length(allMerges{i})-1;
    if( bOutFile)
        fprintf(out_fid, '%d\t%d\t%.3f\t%.3f\t%.3f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t', ...
        i, nSpike(i), meanAmp(i), ...
        meanPos(i), unDetected(i), bestMiss(i), bestFP(i),bestScore(i), autoMiss(i),...
        autoFP(i), autoScore(i), nMerges(i));
        for j = 1:length(allMerges{i})-1
            fprintf( out_fid, '%d,', allMerges{i}(j)-1 );
        end
        fprintf(out_fid,'%d\n',allMerges{i}(length(allMerges{i}))-1);   
    else
        fprintf('%d\t%d\t%.3f\t%.3f\t%.3f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t', ...
        i, nSpike(i), meanAmp(i), ...
        meanPos(i), unDetected(i), bestMiss(i), bestFP(i),bestScore(i), autoMiss(i),...
        autoFP(i), autoScore(i), nMerges(i));
        for j = 1:length(allMerges{i})-1
            fprintf( '%d,', allMerges{i}(j)-1 );
        end
        fprintf('%d\n',allMerges{i}(length(allMerges{i}))-1); 
    end
end

fclose(out_fid);

end