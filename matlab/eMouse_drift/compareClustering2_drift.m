

function [allScores, allFPs, allMisses, allMerges, unDetected, overDetected, GTcluIDs] = compareClustering2(cluGT, resGT, cluTest, resTest, datFilename)
% function compareClustering(cluGT, resGT, cluTest, resTest[, datFilename])
% - clu and res variables are length nSpikes, for ground truth (GT) and for
% the clustering to be evaluated (Test). 


if nargin<5
    datFilename = [];
end

GTcluIDs = unique(cluGT);   %makes a sorted list of the unique labels
testCluIDs = unique(cluTest);
jitter = 6;

%count up spikes for each label in the test set
nSp = zeros(max(testCluIDs), 1);
for j = 1:max(testCluIDs);
    nSp(j) = max(1, sum(cluTest==j));
end
nSp0 = nSp; %array of nSpikes per test cluster

for cGT = 1:length(GTcluIDs)
%     fprintf(1,'ground truth cluster ID = %d (%d spikes)\n', GTcluIDs(cGT), sum(cluGT==GTcluIDs(cGT)));
    
    rGT = int32(resGT(cluGT==GTcluIDs(cGT)));   %set of GT times for this cluster
    
%     S = sparse(numel(rGT), max(testCluIDs));
    % allocate space for counting test cluster assignments to each spike in
    % the set of GT spikes for this cluster
    %"spalloc" specifies that this must be a sparse matrix.
    % creates an M-by-N all zero sparse matrix
    % with room to eventually hold NZMAX nonzeros.
    S = spalloc(numel(rGT), double(max((testCluIDs))), numel(rGT) * 10); 
    % find the initial best match
    mergeIDs = [];
    scores = [];
    falsePos = [];
    missRate = [];
    
    igt = 1;
    
    nSp = nSp0;
    nrGT = numel(rGT);
    flag = false;
    for j = 1:numel(cluTest)            %loop over assigned spikes from clustering
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
        
        if resTest(j)>rGT(igt)-jitter   %note that resTest(j) known to be < rGT(igt) + jitter
            % we found a match, add a tick to the right cluster
%             numMatch(cluTest(j)) = numMatch(cluTest(j)) + 1;
              S(igt, cluTest(j)) = 1;
        end
    end
    numMatch = sum(S,1)';   %array of matched spikes from each test cluster for this GT cluster
    timesMatch = full(sum(S,2)); %array matches per gt time
    udFrac = (sum(timesMatch==0)/nrGT); %fraction of GT spikes matched, independent of cluster assignment (scalar);  
    odFrac = (sum(timesMatch>1)/nrGT);
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
    sc = 1-(fps+misses);        %array of scores for test clusters compared to this GT cluster
    best = find(sc==max(sc),1);
    mergeIDs(end+1) = best;
    scores(end+1) = sc(best);
    falsePos(end+1) = fps(best);
    missRate(end+1) = misses(best);
    
%     fprintf(1, '  found initial best %d: score %.2f (%d spikes, %.2f FP, %.2f miss)\n', ...
%         mergeIDs(1), scores(1), sum(cluTest==mergeIDs(1)), fps(best), misses(best));
    
    S0 = S(:, best);
    nSp = nSp + nSp0(best);
    %attempt merges until 
    %   -there's only one cluster left OR
    %   -the score isn't improving AND the score is nearly = 1
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
        nSp = nSp + nSp0(best);  %merge cluster
                
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
    unDetected(cGT) = udFrac;
    overDetected(cGT) = odFrac;
end    %end of loop over GT cluster IDs

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
