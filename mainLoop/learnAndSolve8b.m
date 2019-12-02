function rez = learnAndSolve8b(rez)
% This is the main optimization. Takes the longest time and uses the GPU heavily.


if ~isfield(rez, 'W') || isempty(rez.W)
    Nbatches = numel(rez.iorig);
    ihalf = ceil(Nbatches/2); % more robust to start the tracking in the middle of the re-ordered batches

    % we learn the templates by going back and forth through some of the data,
    % in the order specified by iorig (determined by batch reordering).
    iorder0 = rez.iorig([ihalf:-1:1 1:ihalf]); % these are absolute batch ids
    rez     = learnTemplates(rez, iorder0);

    fexp = exp(double(nsp0).*log(pm(1:Nfilt)));
    fexp = reshape(fexp, 1,1,[]);
    nsp = nsp * p1 + (1-p1) * double(nsp0);
    dWU = dWU .* fexp + (1-fexp) .* (dWU0./reshape(max(1, double(nsp0)), 1,1, []));

    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    if ibatch==niter-nBatches
        flag_resort   = 0;
        flag_final = 1;

        % final clean up
        [W, U, dWU, mu, nsp, ndrop] = ...
            triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop);

        Nfilt = size(W,2);
        Params(2) = Nfilt;

        [WtW, iList] = getMeWtW(single(W), single(U), Nnearest);

        [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
        iW = int32(squeeze(iW));

        % extract ALL features on the last pass
        Params(13) = 2;

        % different threshold on last pass?
        Params(3) = ops.Th(end);

        rez = memorizeW(rez, W, dWU, U, mu);
        fprintf('memorized middle timepoint \n')
    end

    if ibatch<niter-nBatches %-50
        if rem(ibatch, 5)==1
            % this drops templates
            [W, U, dWU, mu, nsp, ndrop] = ...
                triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop);
        end
        Nfilt = size(W,2);
        Params(2) = Nfilt;

        % this adds templates
%         dWU0 = mexGetSpikes(Params, drez, wPCA);
        [dWU0,cmap] = mexGetSpikes2(Params, drez, wTEMP, iC-1);

        if size(dWU0,3)>0
            dWU0 = double(dWU0);
            dWU0 = reshape(wPCAd * (wPCAd' * dWU0(:,:)), size(dWU0));
            dWU = cat(3, dWU, dWU0);

            W(:,Nfilt + [1:size(dWU0,3)],:) = W0(:,ones(1,size(dWU0,3)),:);

            nsp(Nfilt + [1:size(dWU0,3)]) = ops.minFR * NT/ops.fs;
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
        rez.WA(:,:,:,k) = gather(W);
        rez.UA(:,:,:,k) = gather(U);
        rez.muA(:,k) = gather(mu);

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
        st3(irange,5) = korder;

        fW(:, irange) = gather(featW);

        fWpc(:, :, irange) = gather(featPC);

        ntot = ntot + numel(x0);
    end

    if ibatch==niter-nBatches
        st3 = zeros(1e7, 5);
        rez.WA = zeros(nt0, Nfilt, Nrank,nBatches,  'single');
        rez.UA = zeros(Nchan, Nfilt, Nrank,nBatches,  'single');
        rez.muA = zeros(Nfilt, nBatches,  'single');

        fW  = zeros(Nnearest, 1e7, 'single');
        fWpc = zeros(NchanNear, Nrank, 1e7, 'single');
    end

    if (rem(ibatch, 100)==1)
        fprintf('%2.2f sec, %d / %d batches, %d units, nspks: %2.4f, mu: %2.4f, nst0: %d, merges: %2.4f, %2.4f \n', ...
            toc, ibatch, niter, Nfilt, sum(nsp), median(mu), numel(st0), ndrop)

%         keyboard;

        if ibatch==1
            figHand = figure;
        else
            figure(figHand);
        end

       if ops.fig
           subplot(2,2,1)
           imagesc(W(:,:,1))
           title('Temporal Components')
           xlabel('Unit number');
           ylabel('Time (samples)');

           subplot(2,2,2)
           imagesc(U(:,:,1))
           title('Spatial Components')
           xlabel('Unit number');
           ylabel('Channel number');

           subplot(2,2,3)
           plot(mu)
           ylim([0 100])
           title('Unit Amplitudes')
           xlabel('Unit number');
           ylabel('Amplitude (arb. units)');

           subplot(2,2,4)
           semilogx(1+nsp, mu, '.')
           ylim([0 100])
           xlim([0 100])
           title('Amplitude vs. Spike Count')
           xlabel('Spike Count');
           ylabel('Amplitude (arb. units)');
           drawnow
        end
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
rez.st2 = st3;

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


nKeep = min(Nchan*3,20); % how many PCs to keep
rez.W_a = zeros(nt0 * Nrank, nKeep, Nfilt, 'single');
rez.W_b = zeros(nBatches, nKeep, Nfilt, 'single');
rez.U_a = zeros(Nchan* Nrank, nKeep, Nfilt, 'single');
rez.U_b = zeros(nBatches, nKeep, Nfilt, 'single');
for j = 1:Nfilt
    WA = reshape(rez.WA(:, j, :, :), [], nBatches);
    WA = gpuArray(WA);
    [A, B, C] = svdecon(WA);
    rez.W_a(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.W_b(:,:,j) = gather(C(:, 1:nKeep));

    UA = reshape(rez.UA(:, j, :, :), [], nBatches);
    UA = gpuArray(UA);
    [A, B, C] = svdecon(UA);
    rez.U_a(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.U_b(:,:,j) = gather(C(:, 1:nKeep));
end

rez.ops.fig = 0;
rez         = runTemplates(rez);
