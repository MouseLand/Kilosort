function rez = learnTemplates(rez, iorder)
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
[wTEMP, wPCA]    = extractTemplatesfromSnippets(rez, NrankPC);

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
pmi = exp(-1./linspace(ops.momentum(1), ops.momentum(2), niter));

Nsum = min(Nchan,7); % how many channels to extend out the waveform in mexgetspikes
% lots of parameters passed into the CUDA scripts
Params     = double([NT Nfilt ops.Th(1) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi(1) Nchan NchanNear ops.nt0min 1 Nsum NrankPC ops.Th(1) useStableMode]);

% W0 has to be ordered like this
W0 = permute(double(wPCA), [1 3 2]);

% initialize the list of channels each template lives on
iList = int32(gpuArray(zeros(Nnearest, Nfilt)));

% initialize average number of spikes per batch for each template
nsp = gpuArray.zeros(0,1, 'double');

% this flag starts 0, is set to 1 later
Params(13) = 0;

% kernels for subsample alignment
[Ka, Kb] = getKernels(ops, 10, 1);

p1 = .95; % decay of nsp estimate in each batch

fprintf('Time %3.0fs. Optimizing templates ...\n', toc)

fid = fopen(ops.fproc, 'r');

ndrop = zeros(1,2); % this keeps track of dropped templates for debugging purposes

m0 = ops.minFR * ops.NT/ops.fs; % this is the minimum firing rate that all templates must maintain, or be dropped

for ibatch = 1:niter    
    k = iorder(ibatch); % k is the index of the batch in absolute terms

    % obtained pm for this batch
    Params(9) = pmi(ibatch);
    pm = pmi(ibatch);
    
    % loading a single batch (same as everywhere)
    offset = 2 * ops.Nchan*batchstart(k);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');
    dataRAW = single(gpuArray(dat))/ ops.scaleproc;


    if ibatch==1
       % only on the first batch, we first get a new set of spikes from the residuals,
       % which in this case is the unmodified data because we start with no templates
        [dWU, cmap] = mexGetSpikes2(Params, dataRAW, wTEMP, iC-1); % CUDA function to get spatiotemporal clips from spike detections
        dWU = double(dWU);
        dWU = reshape(wPCAd * (wPCAd' * dWU(:,:)), size(dWU)); % project these into the wPCA waveforms

        W = W0(:,ones(1,size(dWU,3)),:); % initialize the low-rank decomposition with standard waves
        Nfilt = size(W,2); % update the number of filters/templates
        nsp(1:Nfilt) = m0; % initialize the number of spikes for new templates with the minimum allowed value, so it doesn't get thrown back out right away
        Params(2) = Nfilt; % update in the CUDA parameters
    end

    
    % resort the order of the templates according to best peak channel
    % this is important in order to have cohesive memory requests from the GPU RAM
    [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2); % max channel (either positive or negative peak)
    iW = int32(squeeze(iW));
    
    [iW, isort] = sort(iW); % sort by max abs channel
    W = W(:,isort, :); % user ordering to resort all the other template variables
    dWU = dWU(:,:,isort);
    nsp = nsp(isort);
    

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

    if ibatch<niter
        % during the main "learning" phase of fitting a model
        if rem(ibatch, 5)==1
            % this drops templates based on spike rates and/or similarities to other templates
            [W, U, dWU, mu, nsp, ndrop] = ...
                triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop);
        end
        Nfilt = size(W,2); % update the number of filters
        Params(2) = Nfilt;
        
        % this adds new templates if they are detected in the residual
        [dWU0,cmap] = mexGetSpikes2(Params, drez, wTEMP, iC-1);
        
        if size(dWU0,3)>0
            % new templates need to be integrated into the same format as all templates
            dWU0 = double(dWU0);
            dWU0 = reshape(wPCAd * (wPCAd' * dWU0(:,:)), size(dWU0)); % apply PCA for smoothing purposes
            dWU = cat(3, dWU, dWU0);
            
            W(:,Nfilt + [1:size(dWU0,3)],:) = W0(:,ones(1,size(dWU0,3)),:); % initialize temporal components of waveforms
            
            nsp(Nfilt + [1:size(dWU0,3)]) = ops.minFR * NT/ops.fs; % initialize the number of spikes with the minimum allowed
            mu(Nfilt + [1:size(dWU0,3)])  = 10; % initialize the amplitude of this spike with a lowish number
            
            Nfilt = min(ops.Nfilt, size(W,2)); % if the number of filters exceed the maximum allowed, clip it
            Params(2) = Nfilt;
            
            W   = W(:, 1:Nfilt, :); % remove any new filters over the maximum allowed
            dWU = dWU(:, :, 1:Nfilt); % remove any new filters over the maximum allowed
            nsp = nsp(1:Nfilt); % remove any new filters over the maximum allowed
            mu  = mu(1:Nfilt); % remove any new filters over the maximum allowed
        end
    end

        
    if (rem(ibatch, 100)==1)
        % this is some of the relevant diagnostic information to be printed during training
        fprintf('%2.2f sec, %d / %d batches, %d units, nspks: %2.4f, mu: %2.4f, nst0: %d, merges: %2.4f, %2.4f \n', ...
            toc, ibatch, niter, Nfilt, sum(nsp), median(mu), numel(st0), ndrop)

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

% We need to memorize the state of the templates at this timepoint.
% final clean up, triage templates one last time
[W, U, dWU, mu, nsp, ndrop] = ...
    triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop);
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

% save('rez_mid.mat', 'rez');

fprintf('Finished learning templates \n')
%%

