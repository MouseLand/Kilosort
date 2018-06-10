function rez = learnAndSolve10(rez)

ops = rez.ops;


wPCA    = extractPCfromSnippets(rez, 3);
wPCA = gpuArray(wPCA);

ops.wPCA = wPCA;
% wPCA = gpuArray(ops.wPCA(:,1:3)); 

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

% [iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear);
[iC, mask] = getClosestChannels(rez, sigmaMask, NchanNear);


isortbatches = rez.iorig(:);
nhalf = ceil(nBatches/2);

ischedule = [nhalf:nBatches nBatches:-1:nhalf];
i1 = [(nhalf-1):-1:1];
i2 = [nhalf:nBatches];
    
irounds = cat(2, ischedule, i1, i2);

niter   = numel(irounds); 
if irounds(niter - nBatches)~=nhalf
    error('mismatch between number of batches');
end

flag_resort      = 1;

t0 = ceil(rez.ops.trange(1) * ops.fs);    

nInnerIter  = 20;

ThSi        = ops.ThS(1);

pmi = exp(-1./linspace(ops.momentum(1), ops.momentum(2), niter-nBatches));


W0 = permute(wPCA, [1 3 2]);

iList = int32(gpuArray(ones(Nnearest, ops.Nfilt)));

nsp = gpuArray.zeros(0,1, 'single');
sd  = gpuArray.zeros(0,1, 'single');
derr  = gpuArray.zeros(0,1, 'single');
tstate  = gpuArray.zeros(0,1, 'single');
dWU  = gpuArray.zeros(nt0, ops.Nchan,0, 'single');
W  = gpuArray.zeros(nt0, 0, Nrank, 'single');

Params     = double([NT 0 ops.Th(1) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi(1) Nchan NchanNear ThSi(1) 1]);
Params(13) = 0;

p1 = .95; % decay of nsp estimate

fprintf('Time %3.0fs. Optimizing templates ...\n', toc)

fid = fopen(ops.fproc, 'r');

ntot = 0;

for ibatch = 1:niter
    %     k = irounds(ibatch);
    korder = irounds(ibatch);    
    k = isortbatches(korder); 
    
    if ibatch>niter-nBatches && korder==nhalf  
        [W, dWU] = revertW(rez);
        
        Params(2) = size(W,2);
        
        [nsp, sd, derr, tstate] = ...
            removeEndTemplates(size(W,2),nsp, sd, derr, tstate);
        
        fprintf('reverted back to middle timepoint \n')
    end
    
    if ibatch<=niter-nBatches
        Params(9) = pmi(ibatch);
        pm = pmi(ibatch) * gpuArray.ones(1, ops.Nfilt, 'single');
    end
    
    % dat load \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    offset = 2 * ops.Nchan*batchstart(k);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');
    dataRAW = single(gpuArray(dat))/ ops.scaleproc;
    
    if ibatch==1               
        [Params, dWU, W, nsp, sd, derr, tstate] = ...
            getNewTemplates(ops,Params, dWU, W, dataRAW, wPCA, W0, nsp, sd, derr, tstate);
    end
    
    % resort templates by max channel for computational efficiency
    [W, dWU, nsp, sd, derr, iW] = ...
        resortTemplates(nt0min, W, dWU, nsp, sd, derr, tstate);    
    
    % decompose dWU by svd of time and space (61 by 61)
    [W, U, mu] = mexSVDsmall(Params, dWU, W, iC-1, iW-1);
    
    % this needs to change
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan);
      
    [st0, id0, x0, featW, dWU, drez, nsp0, ss0, featPC, sd, derr] = ...
        mexMPnu10(Params, dataRAW, dWU, U, W, mu, iC-1, iW-1, UtU, iList-1, ...
        wPCA, maskU, pm, sd, derr);    
  
    nsp = nsp * p1 + (1-p1) * nsp0;
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    if ibatch==niter-nBatches                
        flag_resort   = 0;
        
        % final clean up
%         [W, U, dWU, mu, sd, derr, nsp, tstate] = ...
%             triageTemplates2(ops, iW, C2C, W, U, dWU, mu, sd ,derr, nsp, tstate);        
        [W, U, dWU, mu, nsp] = ...
            triageTemplates(ops, W, U, dWU, mu, nsp, 1);        
        
        Params(2) = size(W,2); 
        
        % these templates can no longer change!
        tstate(1:size(W,2)) = 0;
        
        [WtW, iList] = getMeWtW(W, U, Nnearest, ops);
        
        [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
        iW = int32(squeeze(iW));
        
        % extract ALL features on the last pass 
        Params(13) = 2;
        
        % different threshold on last pass?
        Params(3) = ops.Th(end);

        rez = memorizeW(rez, W, dWU, U, mu);
        fprintf('memorized middle timepoint \n')
    end
    
    if rem(ibatch, 5)==1
        % this drops templates
%         [W, U, dWU, mu, sd, derr, nsp, tstate] = ...
%             triageTemplates2(ops, iW, C2C, W, U, dWU, mu, sd ,derr, nsp, tstate);
        [W, U, dWU, mu, nsp] = ...
            triageTemplates(ops, W, U, dWU, mu, nsp, 1);        
    end
    Params(2) = size(W,2);
    
    [Params, dWU, W, nsp, mu, sd, derr, tstate] = ...
        getNewTemplates(ops,Params, dWU, W, drez, wPCA, W0, nsp, sd, derr, tstate);        
    
    if ibatch>niter-nBatches
        toff = nt0min + t0 + (NT-ops.ntbuff)*(k-1);        
%         toff = nt0min + t0 + (NT-ops.ntbuff)*(korder-1);        
        
        if k>1 
            toff = toff - ops.ntbuff;            
        end
        
        irange = ntot + [1:numel(x0)];
 
        st3(irange,:) = [toff + double(st0) double(id0+1) double(x0) double(ss0(:,1))];
        fW(:, irange) = gather(featW);        
        fWpc(:, :, irange) = gather(featPC);
        
        ntot = ntot + numel(x0);
    end
    
    if ibatch==niter-nBatches        
        st3 = zeros(1e7, 4);
        fW  = zeros(Nnearest, 1e7, 'single');
        fWpc = zeros(NchanNear, Nrank, 1e7, 'single');
    end    
    
    if rem(ibatch, 100)==1
        make_update(ibatch, niter, nsp, mu, W, U);
    end
    
end

Nfilt = size(rez.W, 2);
ix = st3(1:ntot, 2)<=Nfilt;

fclose(fid);

st3 = st3(ix, :);
fW = fW(:, ix);
fWpc = fWpc(:,:, ix);


fprintf('Time %3.0f. Spike sorted %d spikes. \n', toc, ntot)


[~, isort] = sort(st3(:,1), 'ascend');

fW = fW(:, isort);
fWpc = fWpc(:,:,isort);
st3 = st3(isort, :);

rez.st3 = st3;

rez.simScore = gather(max(WtW, [], 3));

rez.cProj    = fW';
rez.iNeigh   = gather(iList(:, 1:Nfilt));

rez.ops = ops;

% rez.W = cat(1, zeros(nt0 - (ops.nt0-1-nt0min), Nfilt, Nrank), rez.W);
rez.nsp = nsp(1:Nfilt);

rez.cProjPC     = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');

[~, iNch]       = sort(abs(rez.U(:,:,1)), 1, 'descend');
rez.iNeighPC    = gather(iC(:, iW(1:Nfilt)));


% rez.muall = muall;
% rez.Wall = Wall;
% rez.Uall = Uall;
