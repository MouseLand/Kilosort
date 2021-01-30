function rez = learnTemplates2(rez, iorder)
% This is the main optimization. Takes the longest time and uses the GPU heavily.  

rez.ops.fig = getOr(rez.ops, 'fig', 1); % whether to show plots every N batches

% Turn on sorting of spikes before subtracting and averaging in mpnu8
rez.ops.useStableMode = getOr(rez.ops, 'useStableMode', 1);
useStableMode = rez.ops.useStableMode;

NrankPC = 6; % this one is the rank of the PCs, used to detect spikes with threshold crossings
Nrank = 3; % this one is the rank of the templates

rez.ops.LTseed = getOr(rez.ops, 'LTseed', 1);
rng('default'); rng(rez.ops.LTseed);

ops = rez.ops;

% we need PC waveforms, as well as template waveforms
wTEMP = rez.wTEMP;
wPCA  = rez.wPCA; 

% move these to the GPU
wPCA = gpuArray(wPCA(:, 1:Nrank));
wTEMP = gpuArray(wTEMP);
wPCAd = double(wPCA); % convert to double for extra precision
ops.wPCA = gather(wPCA);
ops.wTEMP = gather(wTEMP);
nt0 = ops.nt0;
nt0min  = rez.ops.nt0min;
rez.ops = ops;
nBatches  = rez.temp.Nbatch;
NT  	= ops.NT;
Nfilt 	= ops.Nfilt;
Nchan 	= ops.Nchan;

% two variables for the same thing? number of nearest channels to each primary channel
NchanNear   = min(ops.Nchan, 32);
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
pmi = exp(-1./linspace(ops.momentum(2), ops.momentum(2), niter));

Nsum = min(Nchan,7); % how many channels to extend out the waveform in mexgetspikes
% lots of parameters passed into the CUDA scripts

% initialize the list of channels each template lives on
iList = int32(gpuArray(zeros(Nnearest, Nfilt)));


dWU = gpuArray(rez.dWU);
W = gpuArray(rez.W);
Nfilt = size(W,2); % update the number of filters/templates
% initialize average number of spikes per batch for each template
nsp = gpuArray.zeros(Nfilt,1, 'double');

Params     = double([NT Nfilt ops.Th(1) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi(1) Nchan NchanNear ops.nt0min 1 Nsum NrankPC ops.Th(1) useStableMode]);
Params(2) = Nfilt; % update in the CUDA parameters
% this flag starts 0, is set to 1 later
Params(13) = 0;

% kernels for subsample alignment
[Ka, Kb] = getKernels(ops, 10, 1);

p1 = .95; % decay of nsp estimate in each batch

fprintf('Time %3.0fs. Optimizing templates ...\n', toc)

fid = fopen(ops.fproc, 'r');

%%

for ibatch = 1:niter        
    k = iorder(ibatch); % k is the index of the batch in absolute terms

    % obtained pm for this batch
    Params(9) = pmi(ibatch);
    pm = pmi(ibatch);
    
    % loading a single batch (same as everywhere)
    offset = 2 * ops.Nchan*batchstart(k);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [ops.Nchan NT+ops.ntbuff], '*int16');
    dat = dat';
    dataRAW = single(gpuArray(dat))/ ops.scaleproc;
    Params(1) = size(dataRAW,1);
    
    % resort the order of the templates according to best peak channel
    % this is important in order to have cohesive memory requests from the GPU RAM
    [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2); % max channel (either positive or negative peak)
    iW = int32(squeeze(iW));
    
    % decompose dWU by svd of time and space (via covariance matrix of 61 by 61 samples)
    % this uses a "warm start" by remembering the W from the previous iteration
    [W, U, mu] = mexSVDsmall2(Params, dWU, W, iC-1, iW-1, Ka, Kb);

    % UtU is the gram matrix of the spatial components of the low-rank SVDs
    % it tells us which pairs of templates are likely to "interfere" with each other
    % such as when we subtract off a template
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan); % this needs to change (but I don't know why!)


    % main CUDA function in the whole codebase. does the iterative template matching
    % based on the current templates, gets features for these templates if requested (featW, featPC),
    % gets scores for the template fits to each spike (vexp), outputs the average of
    % waveforms assigned to each cluster (dWU0),
    % and probably a few more things I forget about
    [st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp, errmsg] = ...
        mexMPnu8(Params, dataRAW, single(U), single(W), single(mu), iC-1, iW-1, UtU, iList-1, ...
        wPCA);
    
    
    % errmsg returns 1 if caller requested "stableMode" but mexMPnu8 was
    % compiled without the sorter enabled (i.e. STABLEMODE_ENABLE = false
    % in mexGPUAll). Send an error message to the console just once if this
    % is the case:
    if (ibatch == 1)
        if( (useStableMode == 1) && (errmsg == 1) )
            fprintf( 'useStableMode selected but STABLEMODE not enabled in compiled mexMPnu8.\n' );
        end
    end
    
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
    fexp = exp(double(nsp0).*log(pm));
    fexp = reshape(fexp, 1,1,[]);
    dWU = dWU .* fexp + (1-fexp) .* (dWU0./reshape(max(1, double(nsp0)), 1,1, []));

    % nsp just gets updated according to the fixed factor p1
    nsp = nsp * p1 + (1-p1) * double(nsp0);
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    if (rem(ibatch, 100)==1)
        % this is some of the relevant diagnostic information to be printed during training
        fprintf('%2.2f sec, %d / %d batches, %d units, nspks: %2.4f, mu: %2.4f, nst0: %d \n', ...
            toc, ibatch, niter, Nfilt, sum(nsp), median(mu), numel(st0))

        % these diagnostic figures should be mostly self-explanatory
        if ops.fig
            try
                if ibatch==1
                    figHand = figure;
                else
                    figure(figHand);
                end
                make_fig(W, U, mu, nsp)           
            catch ME
               warning('Error making figure was: %s',ME.message);
            end
        end
    end
end
fclose(fid);
toc

%%
% final covariance matrix between all templates
[WtW, iList] = getMeWtW(single(W), single(U), Nnearest);
% iW is the final channel assigned to each template
[~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
iW = int32(squeeze(iW));
% the similarity score between templates is simply the correlation,
% taken as the max over several consecutive time delays
rez.simScore = gather(max(WtW, [], 3));

rez.iNeighPC    = gather(iC(:, iW));

% the neihboring templates idnices are stored in iNeigh
rez.iNeigh   = gather(iList);

rez = memorizeW(rez, W, dWU, U, mu); % memorize the state of the templates
rez.ops = ops; % update these (only rez comes out of this script)
rez.nsp = nsp;

% save('rez_mid.mat', 'rez');

fprintf('Finished learning templates \n')
%%

