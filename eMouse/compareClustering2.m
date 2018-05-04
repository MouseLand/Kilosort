

function [allScores, allFPs, allMisses, allMerges] = compareClustering2(cluGT, resGT, cluTest, resTest, datFilename)
% function compareClustering(cluGT, resGT, cluTest, resTest[, datFilename])
% - clu and res variables are length nSpikes, for ground truth (GT) and for
% the clustering to be evaluated (Test). 


if nargin<5
    datFilename = [];
end

GTcluIDs = unique(cluGT);
testCluIDs = unique(cluTest);
jitter = 12;

nSp = zeros(max(testCluIDs), 1);
for j = 1:max(testCluIDs);
    nSp(j) = max(1, sum(cluTest==j));
end
nSp0 = nSp;

for cGT = 1:length(GTcluIDs)
%     fprintf(1,'ground truth cluster ID = %d (%d spikes)\n', GTcluIDs(cGT), sum(cluGT==GTcluIDs(cGT)));
    
    rGT = int32(resGT(cluGT==GTcluIDs(cGT)));
    
%     S = sparse(numel(rGT), max(testCluIDs));
    S = spalloc(numel(rGT), max(testCluIDs), numel(rGT) * 10);
    % find the initial best match
    mergeIDs = [];
    scores = [];
    falsePos = [];
    missRate = [];
    
    igt = 1;
    
    nSp = nSp0;
    nrGT = numel(rGT);
    flag = false;
    for j = 1:numel(cluTest)
        while (resTest(j) > rGT(igt) + jitter)
            % the curent spikes is now too large compared to GT, advance the GT
            igt = igt + 1;
            if igt>nrGT
               flag = true;
               break;
            end
        end
        if flag
            break;
        end
        
        if resTest(j)>rGT(igt)-jitter
            % we found a match, add a tick to the right cluster
%             numMatch(cluTest(j)) = numMatch(cluTest(j)) + 1;
              S(igt, cluTest(j)) = 1;
        end
    end
    numMatch = sum(S,1)';
    misses = (nrGT-numMatch)/nrGT; % missed these spikes, as a proportion of the total true spikes
    fps = (nSp-numMatch)./nSp; % number of comparison spikes not near a GT spike, as a proportion of the number of guesses
        %
    %     for cTest = 1:length(testCluIDs)
%         rTest = int32(resTest(cluTest==testCluIDs(cTest)));
%         
%         [miss, fp] = compareSpikeTimes(rTest, rGT);
%         misses(cTest) = miss;
%         fps(cTest) = fp;
%         
%     end
%     
    sc = 1-(fps+misses);
    best = find(sc==max(sc),1);
    mergeIDs(end+1) = best;
    scores(end+1) = sc(best);
    falsePos(end+1) = fps(best);
    missRate(end+1) = misses(best);
    
%     fprintf(1, '  found initial best %d: score %.2f (%d spikes, %.2f FP, %.2f miss)\n', ...
%         mergeIDs(1), scores(1), sum(cluTest==mergeIDs(1)), fps(best), misses(best));
    
    S0 = S(:, best);
    nSp = nSp + nSp0(best);
    while scores(end)>0 && (length(scores)==1 || ( scores(end)>(scores(end-1) + 1*0.01) && scores(end)<=0.99 ))
        % find the best match
        S = bsxfun(@max, S, S0);
        
        numMatch = sum(S,1)';
        misses = (nrGT-numMatch)/nrGT; % missed these spikes, as a proportion of the total true spikes
        fps = (nSp-numMatch)./nSp; % number of comparison spikes not near a GT spike, as a proportion of the number of guesses
        
        sc = 1-(fps+misses);
        best = find(sc==max(sc),1);
        mergeIDs(end+1) = best;
        scores(end+1) = sc(best);
        falsePos(end+1) = fps(best);
        missRate(end+1) = misses(best);
        
%         fprintf(1, '    best merge with %d: score %.2f (%d/%d new/total spikes, %.2f FP, %.2f miss)\n', ...
%             mergeIDs(end), scores(end), nSp0(best), nSp(best), fps(best), misses(best));
        
        S0 = S(:, best);
        nSp = nSp + nSp0(best);
                
    end
    
    if length(scores)==1 || scores(end)>(scores(end-1)+0.01)
        % the last merge did help, so include it
        allMerges{cGT} = mergeIDs(1:end);
        allScores{cGT} = scores(1:end);
        allFPs{cGT} = falsePos(1:end);
        allMisses{cGT} = missRate(1:end);
    else
        % the last merge actually didn't help (or didn't help enough), so
        % exclude it
        allMerges{cGT} = mergeIDs(1:end-1);
        allScores{cGT} = scores(1:end-1);
        allFPs{cGT} = falsePos(1:end-1);
        allMisses{cGT} = missRate(1:end-1);
    end
    
end

initScore = zeros(1, length(GTcluIDs));
finalScore = zeros(1, length(GTcluIDs));
numMerges = zeros(1, length(GTcluIDs));
fprintf(1, '\n\n--Results Summary--\n')
for cGT = 1:length(GTcluIDs)
%     
%      fprintf(1,'ground truth cluster ID = %d (%d spikes)\n', GTcluIDs(cGT), sum(cluGT==GTcluIDs(cGT)));
%      fprintf(1,'  initial score: %.2f\n', allScores{cGT}(1));
%      fprintf(1,'  best score: %.2f (after %d merges)\n', allScores{cGT}(end), length(allScores{cGT})-1);
%      
     initScore(cGT) = allScores{cGT}(1);
     finalScore(cGT) = allScores{cGT}(end);
     numMerges(cGT) = length(allScores{cGT})-1;
end

fprintf(1, 'median initial score: %.2f; median best score: %.2f\n', median(initScore), median(finalScore));
fprintf(1, 'total merges required: %d\n', sum(numMerges));
