function rez = fullMPMU(rez, DATA)

ops = rez.ops;

Nfilt   = ops.Nfilt;
lam =  ones(Nfilt, 1, 'single');
lam(:)    = ops.lam(3);

[W, U, mu, UtU, nu] = decompose_dWU(ops, rez.dWU, ops.Nrank,rez.ops.kcoords);


pm = exp(-ops.momentum(2));
Params = double([ops.NT ops.Nfilt ops.Th(3) ops.maxFR 10 ops.Nchan ops.Nrank pm ops.epu ops.nt0]);

Params(3) = ops.Th(3);
Params(4) = 50000;
Params(5) = 50; 

if ops.GPU
    U0 = gpuArray(U);
else
    U0 = U;
end
%%
nt0     = rez.ops.nt0;
Nrank   = ops.Nrank;
WtW     = zeros(Nfilt,Nfilt,2*nt0-1, 'single');
for i = 1:Nrank
    for j = 1:Nrank
        utu0 = U0(:,:,i)' * U0(:,:,j);
        if ops.GPU
            wtw0 =  gather_try(mexWtW2(Params, W(:,:,i), W(:,:,j), utu0));
        else
            wtw0 =  getWtW2(Params, W(:,:,i), W(:,:,j), utu0);
            wtw0 = permute(wtw0, [2 3 1]);
        end
        WtW = WtW + wtw0;
        clear wtw0 utu0
%         wtw0 = squeeze(wtw(:,i,:,j,:));
        
    end
end

mWtW = max(WtW, [], 3);
WtW = permute(WtW, [3 1 2]);

if ops.GPU
    WtW = gpuArray(WtW);
end
%%
Nbatch_buff = rez.temp.Nbatch_buff;
Nbatch      = rez.temp.Nbatch;
Nchan       = ops.Nchan;
if ~ops.GPU
   fW = rez.fW; % load fft-ed templates 
end
% mWtW = mWtW - diag(diag(mWtW));

% rez.WtW = gather_try(WtW);
%
clear wtw0 utu0 U0
%
clear nspikes2
st3 = [];
rez.st3 = [];

if ops.verbose
   fprintf('Time %3.0fs. Running the final template matching pass...\n', toc) 
end

if Nbatch_buff<Nbatch
    fid = fopen(ops.fproc, 'r');
end
msg = [];

if ~isempty(ops.nNeigh)
    nNeigh    = ops.nNeigh;
    
    rez.cProj = zeros(5e6, nNeigh, 'single');

    % sort pairwise templates
    nsp = sum(rez.nspikes,2);
    vld = single(nsp>100);
    cr    = mWtW .* (vld * vld');
    cr(isnan(cr)) = 0;
    [~, iNgsort] = sort(cr, 1, 'descend');
    
    % save full similarity score
    rez.simScore = cr;
    maskTT = zeros(Nfilt, 'single');
    rez.iNeigh = iNgsort(1:nNeigh, :);
    for i = 1:Nfilt
        maskTT(rez.iNeigh(:,i),i) = 1;
    end
end
if ~isempty(ops.nNeighPC)
    nNeighPC  = ops.nNeighPC;
    load PCspikes
    ixt = round(linspace(1, size(Wi,1), ops.nt0));
    Wi = Wi(ixt, 1:3);
    rez.cProjPC = zeros(5e6, 3*nNeighPC, 'single');
    
    % sort best channels
    [~, iNch]       = sort(abs(U(:,:,1)), 1, 'descend');
    maskPC          = zeros(Nchan, Nfilt, 'single');
    rez.iNeighPC    = iNch(1:nNeighPC, :);
    for i = 1:Nfilt
        maskPC(rez.iNeighPC(:,i),i) = 1;
    end
    maskPC = repmat(maskPC, 3, 1);
end

irun = 0;
i1nt0 = int32([1:nt0])';
%%
LAM = lam .* (20./mu).^2;

NT = ops.NT;
batchstart = 0:NT:NT*(Nbatch-Nbatch_buff);

for ibatch = 1:Nbatch    
    if ibatch>Nbatch_buff
        offset = 2 * ops.Nchan*batchstart(ibatch-Nbatch_buff); % - ioffset;
        fseek(fid, offset, 'bof');
        dat = fread(fid, [NT ops.Nchan], '*int16');
    else
       dat = DATA(:,:,ibatch); 
    end
    if ops.GPU
        dataRAW = gpuArray(dat);
    else
        dataRAW = dat;
    end
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;
    
    % project data in low-dim space
    if ops.GPU
        data    = gpuArray.zeros(NT, Nfilt, Nrank, 'single');
    else
        data   = zeros(NT, Nfilt, Nrank, 'single');
    end
    for irank = 1:Nrank
        data(:,:,irank) = dataRAW * U(:,:,irank);
    end
    data                = reshape(data, NT, Nfilt*Nrank);

    if ops.GPU
        [st, id, x, errC, PCproj] ...
                        = mexMPmuFEAT(Params,data,W,WtW, mu, lam .* (20./mu).^2, nu);
    else
         [st, id, x, errC, PCproj]= cpuMPmuFEAT(Params,data,fW,WtW, mu, lam .* (20./mu).^2, nu, ops);
    end
    
    if ~isempty(st)
        if ~isempty(ops.nNeighPC)
            % PCA coefficients
            inds            = repmat(st', nt0, 1) + repmat(i1nt0, 1, numel(st));
            try  datSp      = dataRAW(inds(:), :);
            catch
                datSp       = dataRAW(inds(:), :);
            end
            datSp           = reshape(datSp, [size(inds) Nchan]);
            coefs           = reshape(Wi' * reshape(datSp, nt0, []), size(Wi,2), numel(st), Nchan);
            coefs           = reshape(permute(coefs, [3 1 2]), [], numel(st));
            coefs           = coefs .* maskPC(:, id+1);
            iCoefs          = reshape(find(maskPC(:, id+1)>0), 3*nNeighPC, []);
            rez.cProjPC(irun + (1:numel(st)), :) = gather_try(coefs(iCoefs)');
        end
        if ~isempty(ops.nNeigh)
            % template coefficients
            % transform coefficients
            PCproj          = bsxfun(@rdivide, ...
                bsxfun(@plus, PCproj, LAM.*mu), sqrt(1+LAM));
            
            PCproj          = maskTT(:, id+1) .* PCproj;
            iPP             = reshape(find(maskTT(:, id+1)>0), nNeigh, []);
            rez.cProj(irun + (1:numel(st)), :) = PCproj(iPP)';
        end
        % increment number of spikes
        irun            = irun + numel(st);
        
        if ibatch==1;
            ioffset         = 0;
        else
            ioffset         = ops.ntbuff;
        end
        st                  = st - ioffset;
        
        %     nspikes2(1:size(W,2)+1, ibatch) = histc(id, 0:1:size(W,2));
        STT = cat(2, ops.nt0min + double(st) +(NT-ops.ntbuff)*(ibatch-1), ...
            double(id)+1, double(x), ibatch*ones(numel(x),1));
        st3             = cat(1, st3, STT);
    end
    if rem(ibatch,100)==1
%         nsort = sort(sum(nspikes2,2), 'descend');
        fprintf(repmat('\b', 1, numel(msg)));
        msg             = sprintf('Time %2.2f, batch %d/%d,  NTOT %d\n', ...
            toc, ibatch,Nbatch, size(st3,1));        
        fprintf(msg);
        
    end
end
%%
[~, isort]      = sort(st3(:,1), 'ascend');
st3             = st3(isort,:);

rez.st3         = st3;
if ~isempty(ops.nNeighPC)
    % re-sort coefficients for projections
    rez.cProjPC(irun+1:end, :)  = [];
    rez.cProjPC                 = reshape(rez.cProjPC, size(rez.cProjPC,1), [], 3);
    rez.cProjPC                 = rez.cProjPC(isort, :,:);
    for ik = 1:Nfilt
        iSp                     = rez.st3(:,2)==ik;
        OneToN                  = 1:nNeighPC;
        [~, isortNeigh]         = sort(rez.iNeighPC(:,ik), 'ascend');
        OneToN(isortNeigh)      = OneToN;
        rez.cProjPC(iSp, :,:)   = rez.cProjPC(iSp, OneToN, :);
    end
    
    rez.cProjPC                 = permute(rez.cProjPC, [1 3 2]);
end
if ~isempty(ops.nNeigh)
    rez.cProj(irun+1:end, :)    = [];
    rez.cProj                   = rez.cProj(isort, :);

    % re-index the template coefficients
    for ik = 1:Nfilt
        iSp                     = rez.st3(:,2)==ik;
        OneToN                  = 1:nNeigh;
        [~, isortNeigh]         = sort(rez.iNeigh(:,ik), 'ascend');
        OneToN(isortNeigh)      = OneToN;
        rez.cProj(iSp, :)       = rez.cProj(iSp, OneToN);
    end
end


%%
% rez.ops             = ops;
rez.W               = W;
rez.U               = U;
rez.mu              = mu;

rez.t2p = [];
for i = 1:Nfilt
    wav0            = W(:,i,1);
    wav0            = my_conv(wav0', .5)';
   [~, itrough]     = min(wav0);
    [~, t2p]        = max(wav0(itrough:end));
    rez.t2p(i,1)    = t2p;
    rez.t2p(i,2)    = itrough;   
end

rez.nbins           = histc(rez.st3(:,2), .5:1:Nfilt+1);

[~, rez.ypos]       = max(rez.U(:,:,1), [], 1);
if Nbatch_buff<Nbatch
    fclose(fid);
end

% center the templates
rez.W               = cat(1, zeros(nt0 - (ops.nt0-1-ops.nt0min), Nfilt, Nrank), rez.W);
rez.WrotInv         = (rez.Wrot/200)^-1;
%%
Urot = U;
for k = 1:size(U,3)
   Urot(:,:,k)  = rez.WrotInv' * Urot(:,:,k);
end
for n = 1:size(U,2)
    rez.Wraw(:,:,n) = mu(n) * sq(Urot(:,n,:)) * sq(rez.W(:,n,:))';
end
%
