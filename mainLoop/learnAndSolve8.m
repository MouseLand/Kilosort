function rez = learnAndSolve8(rez)

ops = rez.ops;


wPCA    = extractPCfromSnippets(rez, 3);
wPCA = gpuArray(wPCA);

% wPCA = gpuArray(ops.wPCA(:,1:3)); 

wPCA(:,1) = - wPCA(:,1) * sign(wPCA(20,1));

rng('default'); rng(1);

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


[iC, mask] = getClosestChannels(rez, sigmaMask, NchanNear);

irounds = [1:nBatches nBatches:-1:1]; 
% irounds = [1:400 400:-1:1 1:400 400:-1:1 1:nBatches]; 
% niter   = numel(irounds); 
niter   = nfullpasses * numel(irounds); 

flag_resort      = 1;
flag_lastpass    = 0;


t0 = ceil(rez.ops.trange(1) * ops.fs);    

nInnerIter  = 20;

ThSi = ops.ThS(1);

pmi = exp(-1./linspace(ops.momentum(1), ops.momentum(2), niter-nBatches));

Params     = double([NT Nfilt ops.Th nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi(1) Nchan NchanNear ThSi(1) 1]);

W0 = permute(wPCA, [1 3 2]);

iList = int32(gpuArray(zeros(Nnearest, Nfilt)));

nsp = gpuArray.zeros(0,1, 'single');

Params(13) = 0;
%%
fprintf('Time %3.0fs. Optimizing templates ...\n', toc)

fid = fopen(ops.fproc, 'r');

ntot = 0;

for ibatch = 1:niter
    %     k = irounds(ibatch);
    k = irounds(rem(ibatch-1, 2*nBatches)+1);
    
    if ibatch<=niter-nBatches
        Params(9) = pmi(ibatch);
%         Params(12) = ThSi(ibatch);
        pm = pmi(ibatch) * gpuArray.ones(1, Nfilt, 'single');
    end
    
    % dat load \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    offset = 2 * ops.Nchan*batchstart(k);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');
    dataRAW = single(gpuArray(dat))/ ops.scaleproc;
    
    if ibatch==1
        dWU = mexGetSpikes(Params, dataRAW, wPCA);
        dWU = reshape(wPCA * (wPCA' * dWU(:,:)), size(dWU));
        W = W0(:,ones(1,size(dWU,3)),:);
        Nfilt = size(W,2);
        nsp(Nfilt) = 0;
        Params(2) = Nfilt;
    end
    
    if flag_resort
        [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
        iW = int32(squeeze(iW));
        
        [iW, isort] = sort(iW);
        W = W(:,isort, :);
        dWU = dWU(:,:,isort);
        nsp = nsp(isort);
    end
    
    % decompose dWU by svd of time and space (61 by 61)
    [W, U, mu] = mexSVDsmall(Params, dWU, W, iC-1, iW-1);
    
    % this needs to change
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan);
    
    if flag_lastpass
        % save the current parameters
        Wall(:,:,:, k)  = gather(W);
        Uall(:,:,:,k)   = gather(U);
        muall(:,k)      = gather(mu);
    end
    
%           pm = exp(-1./max(400, 100 * nsp));
%     pm = exp(-1./(200 * nsp));
%     pm = exp(-1./(200 * gpuArray.ones(1, Nfilt, 'single')));
    
    [st0, id0, x0, featW, dWU, drez, nsp0, ss0, featPC] = ...
        mexMPnu7(Params, dataRAW, dWU, U, W, mu, iC-1, iW-1, UtU, iList-1, ...
        wPCA, maskU, pm);    
    
    nsp = nsp * .95 + .05 * nsp0;
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    if ibatch==niter-nBatches                
        flag_resort   = 0;
        
        % final clean up
        [W, U, dWU, mu, nsp] = triageTemplates(ops, W, U, dWU, mu, nsp, 1);        
        Nfilt = size(W,2);
        Params(2) = Nfilt; 
        
        [WtW, iList] = getMeWtW(W, U, Nnearest);
        
        [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
        iW = int32(squeeze(iW));
        
        % extract ALL features on the last pass
        Params(13) = 2;
        
        Wall  = zeros(nt0, Nfilt, Nrank, nBatches, 'single');
        Uall  = zeros(Nchan, Nfilt,Nrank,  nBatches, 'single');
        muall = zeros(Nfilt, nBatches, 'single');
    end
    
    if ibatch<niter-nBatches-50
        if rem(ibatch, 5)==1    
            % this drops templates
            [W, U, dWU, mu, nsp] = triageTemplates(ops, W, U, dWU, mu, nsp, 1);
        end
        Nfilt = size(W,2);
        Params(2) = Nfilt;
        
        % this adds templates
        dWU0 = mexGetSpikes(Params, drez, wPCA);
        if size(dWU0,3)>0
            dWU0 = reshape(wPCA * (wPCA' * dWU0(:,:)), size(dWU0));
            dWU = cat(3, dWU, dWU0);
            
            W(:,Nfilt + [1:size(dWU0,3)],:) = W0(:,ones(1,size(dWU0,3)),:);
            
            nsp(Nfilt + [1:size(dWU0,3)]) = .05;
            mu(Nfilt + [1:size(dWU0,3)])  = 10;
            
            Nfilt = min(ops.Nfilt, size(W,2));
            Params(2) = Nfilt;
            
            W   = W(:, 1:Nfilt, :);
            dWU = dWU(:, :, 1:Nfilt);
            nsp = nsp(1:Nfilt);
            mu  = mu(1:Nfilt);
        end
        
    end
    
    if ibatch>niter-nBatches
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
        st3(irange,4) = double(ss0(:,1));
        
        fW(:, irange) = gather(featW);        
        
        fWpc(:, :, irange) = gather(featPC);
        
        ntot = ntot + numel(x0);
    end
    
    if ibatch==niter-nBatches        
        flag_lastpass = 1;
        st3 = zeros(1e7, 4);
        fW  = zeros(Nnearest, 1e7, 'single');
        fWpc = zeros(NchanNear, Nrank, 1e7, 'single');
    end
    
    if rem(ibatch, 100)==1
        fprintf('%2.2f sec, %d / %d batches, %d units, nspks: %2.2f, mu: %2.2f \n', ...
            toc, ibatch, niter, Nfilt, median(nsp), median(mu))
        
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

rez.W = gather(W);
rez.W = cat(1, zeros(nt0 - (ops.nt0-1-nt0min), Nfilt, Nrank), rez.W);

rez.U = gather(U);
rez.mu = mu;

nNeighPC        = size(fWpc,1);
rez.cProjPC     = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');

[~, iNch]       = sort(abs(rez.U(:,:,1)), 1, 'descend');
maskPC          = zeros(Nchan, Nfilt, 'single');
rez.iNeighPC    = gather(iC(:, iW));


rez.muall = muall;
rez.Wall = Wall;
rez.Uall = Uall;
