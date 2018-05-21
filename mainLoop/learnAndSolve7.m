ops = rez.ops;

% wPCA    = extractPCfromSnippets(rez, 3);
% wPCA = gpuArray(wPCA);
wPCA = gpuArray(ops.wPCA(:,1:3)); 
wPCA(:,1) = - wPCA(:,1) * sign(wPCA(21,1));

rng('default'); rng(1);

ops.Nfilt = 512;

NchanNear   = 32;
Nnearest    = 32;
nfullpasses = ops.nfullpasses;

sigmaMask  = ops.sigmaMask;


ops.spkTh = -6;  


nt0 = ops.nt0;
nt0min  = ceil(20 * nt0/61);
rez.ops.nt0min  = nt0min;

nBatches  = rez.temp.Nbatch;
NT  	= ops.NT;
batchstart = 0:NT:NT*nBatches;
Nfilt 	= ops.Nfilt; 
ntbuff  = ops.ntbuff;

Nrank   = ops.Nrank;
maxFR 	= ops.maxFR;

Nchan 	= ops.Nchan;


% make iW, a matrix of neighbouring channels for each channel 

[iC, mask] = getClosestChannels(rez, sigmaMask, NchanNear);

irounds = [1:nBatches nBatches:-1:1]; 
Boffset = 0;
niter   = nfullpasses * 2*nBatches + Boffset; 

flag_resort      = 1;
flag_update      = 1;
flag_remember    = 0;
flag_lastpass    = 0;

pmi = exp(-1./linspace(ops.momentum(1), ops.momentum(2), niter-2*nBatches));
ThSi  = linspace(ops.ThS(1),                 ops.ThS(2), niter-2*nBatches);

t0 = ceil(rez.ops.trange(1) * ops.fs);    

nInnerIter  = 20;

Params     = double([NT Nfilt ops.Th nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi(1) Nchan NchanNear ThSi(1) 0]);

W0 = permute(wPCA, [1 3 2]);

iList = int32(gpuArray(zeros(Nnearest, Nfilt)));

st3 = zeros(1e7, 3);

fW  = zeros(Nnearest, 1e7, 'single');

nsp = gpuArray.zeros(0,1, 'single');

fprintf('Time %3.0fs. Optimizing templates ...\n', toc)

fid = fopen(ops.fproc, 'r');

ntot = 0;
%%
for ibatch = 1:niter
    k = irounds(rem(ibatch-1 + Boffset, 2*nBatches)+1);
    
    if ibatch<=niter-2*nBatches
        Params(9) = pmi(ibatch);
        Params(12) = ThSi(ibatch);
    end
    
    if flag_lastpass
        W  = gpuArray(Wall(:,:,:,k));
        U  = gpuArray(Uall(:,:,:,k));
        mu = gpuArray(muall(:,k));
    end
    
    % dat load \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    offset = 2 * ops.Nchan*batchstart(k);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');
    dataRAW = single(gpuArray(dat))/ ops.scaleproc;
    
    if flag_update
        if ibatch==1            
            dWU = mexGetSpikes(Params, dataRAW, wPCA);
            dWU = reshape(wPCA * (wPCA' * dWU(:,:)), size(dWU));
            W = W0(:,ones(1,size(dWU,3)),:);
            Nfilt = size(W,2);            
            nsp(Nfilt) = 0;
        end
        Params(2) = Nfilt; 
        
        % decompose dWU by svd of time and space (61 by 61)
        [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
        iW = int32(squeeze(iW));
        
        if flag_resort
            [iW, isort] = sort(iW);
            W = W(:,isort, :);
            dWU = dWU(:,:,isort);
            nsp = nsp(isort);
        end
        
        [W, U, mu] = mexSVDsmall(Params, dWU, W, iC-1, iW-1);
       
        [W0, nmax] = get_diffW(W);
        nmax(:) = 0;
        
        % this needs to change
        [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan);
    end
    
    % needs an iList for computing features
    [st0, id0, x0, featW, dWU, drez, nsp0, ss0] = mexMPnu7(Params, dataRAW, dWU, ...
        U, W, mu, iC-1, iW-1, UtU, iList-1, wPCA, maskU);    
    
    nsp = nsp * .95 + .05 * nsp0;
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    if flag_remember && ~flag_lastpass
        % save the current parameters
        Wall(:,:,:, k)  = gather(W);
        Uall(:,:,:,k)   = gather(U);
        muall(:,k)      = gather(mu);
    end
    
    if ibatch==niter-2*nBatches
        flag_remember = 1;
        flag_resort   = 0;
        
        [WtW, iList] = getMeWtW(W, U, Nnearest);
        
        % extract features on the last pass
        Params(13) = 1;
        
        % don't discard bad spikes on last pass
        Params(12) = 1e3;
       
        
        Wall  = zeros(nt0, Nfilt, Nrank, nBatches, 'single');
        Uall  = zeros(Nchan, Nfilt,Nrank,  nBatches, 'single');
        muall = zeros(Nfilt, nBatches, 'single');
    end
    
    if ~flag_remember && Nfilt<512  && rem(ibatch, 5)==0     
        [W, U, dWU, mu, nsp] = triageTemplates(ops, W, U, dWU, mu, nsp, 1);
        
        Nfilt = size(W,2);
        
        if Nfilt<ops.Nfilt
            dWU0 = mexGetSpikes(Params, drez, wPCA);
            if size(dWU0,3)>0
                dWU0 = reshape(wPCA * (wPCA' * dWU0(:,:)), size(dWU0));
                dWU = cat(3, dWU, dWU0);
                
                W(:,Nfilt + [1:size(dWU0,3)],:) = W0(:,ones(1,size(dWU0,3)),:);
                
                Nfilt = min(512, size(W,2));
                
                W = W(:, 1:Nfilt, :);
                dWU = dWU(:, :, 1:Nfilt);
                nsp(Nfilt) = 0;
                mu(Nfilt)  = 0;                
            end
        end
   end
    
    
    if flag_lastpass
        ioffset         = ops.ntbuff;
        if k==1
            ioffset         = 0;
        end
        toff = nt0min + t0 -ioffset + (NT-ops.ntbuff)*(k-1);
        st = toff + double(st0);
        irange = ntot + [1:numel(x0)];
        st3(irange,1) = double(st);
        st3(irange,2) = double(id0+1);
        st3(irange,3) = double(x0);
        fW(:, irange) = gather(featW);
        ntot = ntot + numel(x0);
    end
    
    if ibatch==niter-nBatches
        flag_lastpass = 1;
        flag_update = 0;
    end
    if ibatch==100
        flag_update = 1;
        flag_resort = 1;
    end
    
    if rem(ibatch, 100)==1
       fprintf('%2.2f sec, %d batches \n', toc, ibatch) 
       figure(1)
       subplot(2,2,1)
       imagesc(W(:,:,1))
       
       subplot(2,2,2)
       imagesc(U(:,:,1))
       
       subplot(2,2,3)
       plot(mu)
       
       subplot(2,2,4)
       semilogx(1+nsp, mu, '.')
       
       drawnow
    end
end


fclose(fid);

toc

st3 = st3(1:ntot, :);
fW = fW(:, 1:ntot);

ntot


[~, isort] = sort(st3(:,1), 'ascend');

rez.st3 = st3(isort, :);

rez.simScore = gather(max(WtW, [], 3));

rez.cProj    = fW(:, isort)';
rez.iNeigh   = gather(iList);

rez.ops = ops;

rez.W = Wall(:,:,:, round(nBatches/2));
rez.W = cat(1, zeros(nt0 - (ops.nt0-1-nt0min), Nfilt, Nrank), rez.W);

rez.U = Uall(:,:,:,round(nBatches/2));
rez.mu = muall(:,round(nBatches/2));

nNeighPC        = ops.nNeighPC;
rez.cProjPC     = zeros(size(st3,1), 3, nNeighPC, 'single');

[~, iNch]       = sort(abs(rez.U(:,:,1)), 1, 'descend');
maskPC          = zeros(Nchan, Nfilt, 'single');
rez.iNeighPC    = iNch(1:nNeighPC, :);

% mkdir(savePath)
% rezToPhy0(rez, savePath);

