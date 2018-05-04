function [iBatch, nBatches] = makeBatches(nS, nSamples)

nBatches = ceil(nS/nSamples);
for j = 1:nBatches
   iBatch{j} = nSamples * (j-1) + [1:nSamples]; 
end
iBatch{nBatches}(iBatch{nBatches}>nS) = [];
