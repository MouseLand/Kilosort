function     [W, dWU, isplit] = ...
    halfwaySplits(rez, Params, W, U, mu, dWU, sig,  fid, korder)


ops = rez.ops;

wPCA = gpuArray(ops.wPCA);
wPCA(:,1) = - wPCA(:,1) * sign(wPCA(20,1));

rng('default'); rng(1);

NchanNear   = 32;
Nnearest    = 32;

sigmaMask  = ops.sigmaMask;
ops.spkTh = -6;  

nt0 = ops.nt0;
nt0min  = ceil(20 * nt0/61);
rez.ops.nt0min  = nt0min;

nBatches  = rez.temp.Nbatch;
NT  	= ops.NT;
batchstart = 0:NT:NT*nBatches;

Nrank   = 3; %ops.Nrank;
Nchan 	= ops.Nchan;

% [iC, mask] = getClosestChannels(rez, sigmaMask, NchanNear);
[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear);

isortbatches = rez.iorig(:);

i1 = korder + [0:50];
i2 = korder - [1:50];
    
irounds = cat(2, i1, i2);
niter   = numel(irounds); 

nInnerIter  = 20;
ThSi = ops.ThS(1);

Nfilt = size(W,2);

pmi = exp(-1./ops.momentum(2));
pm = pmi * gpuArray.ones(1, Nfilt, 'single');

Params     = double([NT Nfilt ops.Th(end) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi(1) Nchan NchanNear ThSi(1) 2]);

[WtW, iList] = getMeWtW(W, U, Nnearest);

% kernels to align the waveofrm by
[Ka, Kb] = getKernels(ops, 10, 1);

p1 = .95; % decay of nsp estimate

fprintf('Time %3.0fs. Extracting features for halfway split ...\n', toc)

ntot = 0;

nsp   = gpuArray.zeros(Nfilt,1, 'single');
dnext = gpuArray.zeros(Nfilt,1, 'single');

st3 = zeros(1e7, 4);
fW  = zeros(Nnearest, 1e7, 'single');
fWpc = zeros(NchanNear, Nrank, 1e7, 'single');

[~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
iW = int32(squeeze(iW));


% remember where we started
rez = memorizeW(rez, W, dWU, U, mu,sig);

for ibatch = 1:niter
    %     k = irounds(ibatch);
    korder = irounds(ibatch);    
    k = isortbatches(korder); 
    
    if ibatch==numel(i1)+1
        [W, dWU, sig] = revertW(rez);
        fprintf('reverted back to middle timepoint \n')
    end
    
    % dat load \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\    
    offset = 2 * ops.Nchan*batchstart(k);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');
    dataRAW = single(gpuArray(dat))/ ops.scaleproc;
    
    % decompose dWU by svd of time and space (61 by 61)
    [W, U, mu] = mexSVDsmall2(Params, dWU, W, iC-1, iW-1, Ka, Kb);
    
    % this needs to change
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan);
    
    [st0, id0, x0, featW, dWU, drez, nsp0, ss0, featPC, sig, dnext] = ...
        mexMPnu8(Params, dataRAW, dWU, U, W, mu, iC-1, iW-1, UtU, iList-1, ...
        wPCA, maskU, pm, sig, dnext);
    
    nsp = nsp * p1 + (1-p1) * nsp0;
    
    toff = nt0min + (NT-ops.ntbuff)*(k-1);
    st = toff + double(st0);
    irange = ntot + [1:numel(x0)];
    st3(irange,1) = double(st);
    st3(irange,2) = double(id0+1);
    st3(irange,3) = double(x0);
    st3(irange,4) = double(ss0(:,1));
    
    fW(:, irange) = gather(featW);
    
    fWpc(:, :, irange) = gather(featPC);
    
    ntot = ntot + numel(x0);
end


toc

st3 = st3(1:ntot, :);
fW = fW(:, 1:ntot);
fWpc = fWpc(:,:, 1:ntot);

ntot

[~, isort] = sort(st3(:,1), 'ascend');

fW = fW(:, isort);
fWpc = fWpc(:,:,isort);
st3 = st3(isort, :);

rez.st3 = st3;

rez.simScore = gather(max(WtW, [], 3));

rez.cProj    = fW';
rez.iNeigh   = gather(iList);

rez.ops = ops;

rez.nsp = nsp;

rez.cProjPC     = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');

rez.iNeighPC    = gather(iC(:, iW));

rez1 = splitAllClusters(rez);

% I can get W and dWU from the splits

% I need to replicate nsp, dnext, sig from the originals
W = rez1.W;
dWU = rez1.dWU;
isplit = rez1.isplit;

