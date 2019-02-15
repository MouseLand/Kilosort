function rez = learnAndSolve8b(rez)

ops = rez.ops;

NrankPC = 6;  
Nrank = 3;
[wTEMP, wPCA]    = extractTemplatesfromSnippets(rez, NrankPC);
% wPCA    = extractPCfromSnippets(rez, Nrank);
% wPCA(:,1) = - wPCA(:,1) * sign(wPCA(20,1));

wPCA = gpuArray(wPCA(:, 1:Nrank));
wTEMP = gpuArray(wTEMP);

ops.wPCA = gather(wPCA);
ops.wTEMP = gather(wTEMP);
rez.ops = ops;

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
Nfilt 	= ops.Nfilt;

Nchan 	= ops.Nchan;

% [iC, mask] = getClosestChannels(rez, sigmaMask, NchanNear);
[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear);

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
%
flag_resort      = 1;

t0 = ceil(rez.ops.trange(1) * ops.fs);

nInnerIter  = 60;

ThSi = ops.ThS(1);

pmi = exp(-1./linspace(ops.momentum(1), ops.momentum(2), niter-nBatches));

Nsum = 7;
Params     = double([NT Nfilt ops.Th(1) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi(1) Nchan NchanNear ThSi(1) 1 Nsum NrankPC]);

W0 = permute(wPCA, [1 3 2]);

iList = int32(gpuArray(zeros(Nnearest, Nfilt)));

nsp = gpuArray.zeros(0,1, 'single');
sig = gpuArray.zeros(0,1, 'single');
dnext = gpuArray.zeros(0,1, 'single');
damp = gpuArray.zeros(0,100, 'single');

Params(13) = 0;

[Ka, Kb] = getKernels(ops, 10, 1);

p1 = .95; % decay of nsp estimate

fprintf('Time %3.0fs. Optimizing templates ...\n', toc)

fid = fopen(ops.fproc, 'r');

ntot = 0;
ndrop = zeros(1,3);

m0 = ops.minFR * ops.NT/ops.fs;

for ibatch = 1:niter    
    %     k = irounds(ibatch);
    korder = irounds(ibatch);
    k = isortbatches(korder);

    if ibatch>niter-nBatches && korder==nhalf
        [W, dWU, sig] = revertW(rez);
        fprintf('reverted back to middle timepoint \n')
    end

    if ibatch<=niter-nBatches
        Params(9) = pmi(ibatch);
        pm = pmi(ibatch) * gpuArray.ones(1, Nfilt, 'single');
    end

    % dat load \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    offset = 2 * ops.Nchan*batchstart(k);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');
    dataRAW = single(gpuArray(dat))/ ops.scaleproc;

    if ibatch==1    
%         keyboard;
        [dWU, cmap] = mexGetSpikes2(Params, dataRAW, wTEMP, iC-1);
%         dWU = mexGetSpikesOld(Params, dataRAW, wPCA);
        dWU = reshape(wPCA * (wPCA' * dWU(:,:)), size(dWU));
        
        W = W0(:,ones(1,size(dWU,3)),:);
        Nfilt = size(W,2);
        nsp(1:Nfilt) = m0;
        sig(1:Nfilt) = 5^2;
        damp(1:Nfilt,:) = 0;
        dnext(1:Nfilt) = 20^2; % changed from 5^2
        Params(2) = Nfilt;
    end

    if flag_resort
        [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
        iW = int32(squeeze(iW));

        [iW, isort] = sort(iW);
        W = W(:,isort, :);
        dWU = dWU(:,:,isort);
        nsp = nsp(isort);
        sig = sig(isort);
        dnext = dnext(isort);
        damp = damp(isort,:);
    end

    % decompose dWU by svd of time and space (61 by 61)
    [W, U, mu] = mexSVDsmall2(Params, dWU, W, iC-1, iW-1, Ka, Kb);

    % this needs to change
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan);

    [st0, id0, x0, featW, dWU, drez, nsp0, featPC, sig, dnext, damp, vexp] = ...
        mexMPnu8(Params, dataRAW, dWU, U, W, mu, iC-1, iW-1, UtU, iList-1, ...
        wPCA, maskU, pm, sig, dnext, damp);

    damp = damp * p1 + (1-p1) * damp;
    nsp = nsp * p1 + (1-p1) * nsp0;

    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    if ibatch==niter-nBatches
        flag_resort   = 0;

        % final clean up
        [W, U, dWU, mu, nsp, sig, dnext,damp, ndrop] = ...
            triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, sig, dnext, damp, ndrop);

        Nfilt = size(W,2);
        Params(2) = Nfilt;

        [WtW, iList] = getMeWtW(W, U, Nnearest);

        [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
        iW = int32(squeeze(iW));

        % extract ALL features on the last pass
        Params(13) = 2;

        % different threshold on last pass?
        Params(3) = ops.Th(end);

        rez = memorizeW(rez, W, dWU, U, mu, sig);
        fprintf('memorized middle timepoint \n')
    end

    if ibatch<niter-nBatches %-50
        if rem(ibatch, 5)==1
            % this drops templates
            [W, U, dWU, mu, nsp, sig, dnext, damp, ndrop] = ...
                triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, sig, dnext, damp, ndrop);
        end
        Nfilt = size(W,2);
        Params(2) = Nfilt;

        % this adds templates        
%         dWU0 = mexGetSpikes(Params, drez, wPCA, iC-1);
        [dWU0,cmap] = mexGetSpikes2(Params, drez, wTEMP, iC-1);
        
        if size(dWU0,3)>0    
            dWU0 = reshape(wPCA * (wPCA' * dWU0(:,:)), size(dWU0));
            dWU = cat(3, dWU, dWU0);

            W(:,Nfilt + [1:size(dWU0,3)],:) = W0(:,ones(1,size(dWU0,3)),:);

            nsp(Nfilt + [1:size(dWU0,3)]) = ops.minFR * NT/ops.fs;
            mu(Nfilt + [1:size(dWU0,3)])  = 10;
            sig(Nfilt + [1:size(dWU0,3)])  = 5^2;
            dnext(Nfilt + [1:size(dWU0,3)])  = 20^2; % changed from 5^2
            damp(Nfilt + [1:size(dWU0,3)], :)  = 0; % changed from 5^2

            Nfilt = min(ops.Nfilt, size(W,2));
            Params(2) = Nfilt;

            W   = W(:, 1:Nfilt, :);
            dWU = dWU(:, :, 1:Nfilt);
            nsp = nsp(1:Nfilt);
            mu  = mu(1:Nfilt);
            sig  = sig(1:Nfilt);
            dnext  = dnext(1:Nfilt);
            damp = damp(1:Nfilt, :);
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
        
        if ntot+numel(x0)>size(st3,1)
           fW(:, 2*size(st3,1))    = 0;
           fWpc(:,:,2*size(st3,1)) = 0;
           st3(2*size(st3,1), 1)   = 0;
        end
        
        st3(irange,1) = double(st);
        st3(irange,2) = double(id0+1);
        st3(irange,3) = double(x0);        
        st3(irange,4) = double(vexp);

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
        fprintf('%2.2f sec, %d / %d batches, %d units, nspks: %2.2f, mu: %2.2f, nst0: %d, drop: %2.2f, %2.2f, %2.2f \n', ...
            toc, ibatch, niter, Nfilt, sum(nsp), median(mu), numel(st0), ndrop)

       figure(2)
       subplot(2,2,1)
       imagesc(W(:,:,1))

       subplot(2,2,2)
       imagesc(U(:,:,1))

       subplot(2,2,3)
       plot(mu)
       ylim([0 100])

       subplot(2,2,4)
       semilogx(1+nsp, mu, '.')
       ylim([0 100])
       xlim([0 100])

       drawnow
    end
end


fclose(fid);

toc

st3 = st3(1:ntot, :);
fW = fW(:, 1:ntot);
fWpc = fWpc(:,:, 1:ntot);

ntot

% [~, isort] = sort(st3(:,1), 'ascend');
% fW = fW(:, isort);
% fWpc = fWpc(:,:,isort);
% st3 = st3(isort, :);

rez.st3 = st3;

rez.simScore = gather(max(WtW, [], 3));

rez.cProj    = fW';
rez.iNeigh   = gather(iList);

rez.ops = ops;

rez.nsp = nsp;

% nNeighPC        = size(fWpc,1);
rez.cProjPC     = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');

% [~, iNch]       = sort(abs(rez.U(:,:,1)), 1, 'descend');
% maskPC          = zeros(Nchan, Nfilt, 'single');
rez.iNeighPC    = gather(iC(:, iW));
