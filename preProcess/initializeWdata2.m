function [W, mu, Wheights, irand] = initializeWdata2(call, uprojDAT, Nchan, nPCs, Nfilt, iC)
% this function initializes cluster means for the fast kmeans per batch
% call are time indices for the spikes
% uprojDAT are features projections (Nfeatures by Nspikes)
% some more parameters need to be passed in from the main workspace

%get a set of Nfilt unique spike indices
allSpike = randperm(size(uprojDAT,2));
irand = allSpike(1:Nfilt);
%fprintf( 'in initializeWdata, Nfilt = %d, num unique spikes = %d\n', Nfilt, numel(unique(irand)));
%irand = ceil(rand(Nfilt,1) * size(uprojDAT,2)); % pick random spikes from the sample
% irand = 1:Nfilt;

W = gpuArray.zeros(nPCs, Nchan, Nfilt, 'single');

for t = 1:Nfilt
    ich = iC(:, call(irand(t))); % the channels on which this spike lives
    W(:, ich, t) = reshape(uprojDAT(:, irand(t)), nPCs, []); % for each selected spike, get its features
end
W = reshape(W, [], Nfilt);
%W = W + .001 * gpuArray.randn(size(W), 'single'); % add small amount of noise in case we accidentally picked the same spike twice

mu = sum(W.^2,1).^.5; % get the mean of the template
W = W./(1e-5 + mu); % and normalize the template

W = reshape(W, nPCs, Nchan, Nfilt);
nW = sq(W(1, :, :).^2); % squared amplitude of the first PC feture
W = reshape(W, nPCs * Nchan, Nfilt);

[~, Wheights] = max(nW,[], 1); % determine biggest channel according to the amplitude of the first PC
