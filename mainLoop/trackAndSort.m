function [rez, st3, fWpc] = trackAndSort(rez, varargin)

if ~isempty(varargin)
   iorder = varargin{1};
else
    iorder = 1:rez.ops.Nbatch;
end
% This is the extraction phase of the optimization. 
% iorder is the order in which to traverse the batches

% iorder = 1:rez.ops.Nbatch;

% Turn on sorting of spikes before subtracting and averaging in mpnu8
rez.ops.useStableMode = getOr(rez.ops, 'useStableMode', 1);
useStableMode = rez.ops.useStableMode;

ops = rez.ops;

% revert to the saved templates
W = gpuArray(rez.W);
U = gpuArray(rez.U);
mu = gpuArray(rez.mu);

Nfilt 	= size(W,2);
nt0 = ops.nt0;
Nchan 	= ops.Nchan;

dWU = gpuArray.zeros(nt0, Nchan, Nfilt, 'double');
for j = 1:Nfilt
    dWU(:,:,j) = mu(j) * squeeze(W(:, j, :)) * squeeze(U(:, j, :))';
end


ops.fig = getOr(ops, 'fig', 1); % whether to show plots every N batches

NrankPC = 6; % this one is the rank of the PCs, used to detect spikes with threshold crossings
Nrank   = 3; % this one is the rank of the templates
rng('default'); rng(1); % initializing random number generator

% move these to the GPU
wPCA = gpuArray(ops.wPCA);

nt0min  = rez.ops.nt0min;
rez.ops = ops;
nBatches  = rez.temp.Nbatch;
NT  	= ops.NT;


% two variables for the same thing? number of nearest channels to each primary channel
NchanNear   = min(ops.Nchan, 16);
Nnearest    = min(ops.Nchan, 32);

% decay of gaussian spatial mask centered on a channel
sigmaMask  = ops.sigmaMask;

% spike threshold for finding missed spikes in residuals
ops.spkTh = -6; % why am I overwriting this here?

batchstart = 0:NT:NT*nBatches;

% find the closest NchanNear channels, and the masks for those channels
[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear);

niter   = numel(iorder);

% this is the absolute temporal offset in seconds corresponding to the start of the
% spike sorted time segment
t0 = ceil(rez.ops.trange(1) * ops.fs);

nInnerIter  = 60; % this is for SVD for the power iteration

% schedule of learning rates for the model fitting part
% starts small and goes high, it corresponds approximately to the number of spikes
% from the past that were averaged to give rise to the current template
pm = exp(-1/400);

Nsum = min(Nchan,7); % how many channels to extend out the waveform in mexgetspikes
% lots of parameters passed into the CUDA scripts
Params     = double([NT Nfilt ops.Th(1) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pm Nchan NchanNear ops.nt0min 1 Nsum NrankPC ops.Th(1) useStableMode]);

% initialize average number of spikes per batch for each template
nsp = gpuArray.zeros(Nfilt,1, 'int32');

% extract ALL features on the last pass
Params(13) = 2; % this is a flag to output features (PC and template features)

% different threshold on last pass?
Params(3) = ops.Th(end); % usually the threshold is much lower on the last pass

% kernels for subsample alignment
[Ka, Kb] = getKernels(ops, 10, 1);

p1 = .95; % decay of nsp estimate in each batch

% the list of channels each template lives on
% also, covariance matrix between templates
[~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
iW = int32(squeeze(iW));
[WtW, iList] = getMeWtW(single(W), single(U), Nnearest);

fprintf('Time %3.0fs. Final spike extraction ...\n', toc)

fid = fopen(ops.fproc, 'r');

% allocate variables for collecting results
st3 = zeros(1e7, 5); % this holds spike times, clusters and other info per spike
ntot = 0;


% these ones store features per spike
fW  = zeros(Nnearest, 1e7, 'single'); % Nnearest is the number of nearest templates to store features for
fWpc = zeros(NchanNear, 2*Nrank, 1e7, 'single'); % NchanNear is the number of nearest channels to take PC features from


dWU1 = dWU;

[UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan); % this needs to change (but I don't know why!)
    
for ibatch = 1:niter    
    k = iorder(ibatch); % k is the index of the batch in absolute terms
    
    % loading a single batch (same as everywhere)
    offset = 2 * ops.Nchan*batchstart(k);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [ops.Nchan NT + ops.ntbuff], '*int16');
    dat = dat';
    dataRAW = single(gpuArray(dat))/ ops.scaleproc;
    Params(1) = size(dataRAW,1);
    
    % decompose dWU by svd of time and space (via covariance matrix of 61 by 61 samples)
    % this uses a "warm start" by remembering the W from the previous
    % iteration
     
    % we don't need to update this anymore on every iteraton....
%     [W, U, mu] = mexSVDsmall2(Params, dWU, W, iC-1, iW-1, Ka, Kb);
    
    % UtU is the gram matrix of the spatial components of the low-rank SVDs
    % it tells us which pairs of templates are likely to "interfere" with each other
    % such as when we subtract off a template
%     [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan); % this needs to change (but I don't know why!)%     
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    % main CUDA function in the whole codebase. does the iterative template matching
    % based on the current templates, gets features for these templates if requested (featW, featPC),
    % gets scores for the template fits to each spike (vexp), outputs the average of
    % waveforms assigned to each cluster (dWU0),
    % and probably a few more things I forget about
    
    [st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp, errmsg] = ...
        mexMPnu8(Params, dataRAW, single(U), single(W), single(mu), iC-1, iW-1, UtU, iList-1, ...
        wPCA);
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    % Sometimes nsp can get transposed (think this has to do with it being
    % a single element in one iteration, to which elements are added
    % nsp, nsp0, and pm must all be row vectors (Nfilt x 1), so force nsp
    % to be a row vector.
    [nsprow, nspcol] = size(nsp);
    if nsprow<nspcol
        nsp = nsp';
    end
    
    % updates the templates as a running average weighted by recency
    % since some clusters have different number of spikes, we need to apply the
    % exp(pm) factor several times, and fexp is the resulting update factor
    % for each template

    dWU1 = dWU1  + dWU0;
    nsp = nsp + nsp0;
    
    % nsp just gets updated according to the fixed factor p1
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    % during the final extraction pass, this keesp track of all spikes and features
    
    % we memorize the spatio-temporal decomposition of the waveforms at this batch
    % this is currently only used in the GUI to provide an accurate reconstruction
    % of the raw data at this time
    
    % we carefully assign the correct absolute times to spikes found in this batch
    toff = nt0min + t0 + NT*(k-1);
    st = toff + double(st0);
    
    irange = ntot + [1:numel(x0)]; % spikes and features go into these indices
    
    if ntot+numel(x0)>size(st3,1)
        % if we exceed the original allocated memory, double the allocated sizes
        fW(:, 2*size(st3,1))    = 0;
        fWpc(:,:,2*size(st3,1)) = 0;
        st3(2*size(st3,1), 1)   = 0;
    end
    
    st3(irange,1) = double(st); % spike times
    st3(irange,2) = double(id0+1); % spike clusters (1-indexing)
    st3(irange,3) = double(x0); % template amplitudes
    st3(irange,4) = double(vexp); % residual variance of this spike
    st3(irange,5) = ibatch; % batch from which this spike was found
    
    fW(:, irange) = gather(featW); % template features for this batch
    fWpc(:, :, irange) = gather(featPC); % PC features
    
    ntot = ntot + numel(x0); % keeps track of total number of spikes so far
    
    
    if (rem(ibatch, 100)==1)
        % this is some of the relevant diagnostic information to be printed during training
        fprintf('%2.2f sec, %d / %d batches, %d units, nspks: %d, mu: %2.4f, nst0: %d \n', ...
            toc, ibatch, niter, Nfilt, ntot, median(mu), numel(st0))
        
        % these diagnostic figures should be mostly self-explanatory
        if ops.fig
            if ibatch==1
                figHand = figure;
            else
                figure(figHand);
            end            
            
            make_fig(W, U, mu, nsp)
        end
    end
end
fclose(fid);
toc

% discards the unused portion of the arrays
st3 = st3(1:ntot, :);
fW = fW(:, 1:ntot);
fWpc = fWpc(:,:, 1:ntot);

rez.dWU = dWU1 ./ (1e-10 + single(reshape(nsp, [1,1,Nfilt])));
rez.nsp = nsp;

rez.iC = iC;

fWpc = permute(fWpc, [3, 2, 1]);

%%
