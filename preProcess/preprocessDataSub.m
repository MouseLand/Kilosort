function [rez, DATA] = preprocessDataSub(ops)
tic;
ops.nt0 	= getOr(ops, {'nt0'}, 61);

[chanMap, xc, yc, kcoords, NchanTOTdefault] = loadChanMap(ops.chanMap);
ops.Nchan = numel(chanMap);
ops.NchanTOT = getOr(ops, 'NchanTOT', NchanTOTdefault);

NchanTOT = ops.NchanTOT;
NT       = ops.NT ;

rez.ops         = ops;
rez.xc = xc;
rez.yc = yc;

rez.xcoords = xc;
rez.ycoords = yc;

% rez.connected   = connected;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords; 

d = dir(ops.fbinary);
nTimepoints = floor(d.bytes/NchanTOT/2);

rez.ops.tstart = ceil(ops.trange(1) * ops.fs); 
rez.ops.tend   = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); 

rez.ops.sampsToRead = rez.ops.tend-rez.ops.tstart; 

NTbuff      = NT + 4*ops.ntbuff;
Nbatch      = ceil(rez.ops.sampsToRead /(NT-ops.ntbuff));

% by how many bytes to offset all the batches
twind = rez.ops.tstart * NchanTOT*2;

%% load data into patches, filter, compute covariance
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
end

fprintf('Time %3.0fs. Loading raw data... \n', toc);
fid = fopen(ops.fbinary, 'r');
Nchan = rez.ops.Nchan;
if ops.GPU
    CC = gpuArray.zeros( Nchan,  Nchan, 'single');
else
    CC = zeros( Nchan,  Nchan, 'single');
end

if ops.useRAM
    DATA = zeros(NT, rez.ops.Nchan, Nbatch, 'int16');
else
    DATA = [];
end

ibatch = 1;
while ibatch<=Nbatch    
    offset = max(0, twind + 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    fseek(fid, offset, 'bof');
    buff = fread(fid, [NchanTOT NTbuff], '*int16');
        
    if isempty(buff)
        break;
    end
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end
    if ops.GPU
        dataRAW = gpuArray(buff);
    else
        dataRAW = buff;
    end
    dataRAW = dataRAW';
    dataRAW = single(dataRAW);
    dataRAW = dataRAW(:, chanMap);
    
    % subtract the mean from each channel
    dataRAW = dataRAW - mean(dataRAW, 1);
    
    datr = filter(b1, a1, dataRAW);
    datr = flipud(datr);
    datr = filter(b1, a1, datr);
    datr = flipud(datr);

    % CAR, common average referencing by median
    if getOr(ops, 'CAR', 1)
        datr = datr - median(datr, 2);
    end
    
    CC        = CC + (datr' * datr)/NT;    
    
    ibatch = ibatch + ops.nSkipCov;
end
CC = CC / ceil((Nbatch-1)/ops.nSkipCov);

fclose(fid);
fprintf('Time %3.0fs. Channel-whitening filters computed. \n', toc);

if ops.whiteningRange<Inf
    ops.whiteningRange = min(ops.whiteningRange, Nchan);
    Wrot = whiteningLocal(gather_try(CC), yc, xc, ops.whiteningRange);
else
    [E, D] 	= svd(CC);
    D       = diag(D);
    eps 	= 1e-6;
    Wrot 	= E * diag(1./(D + eps).^.5) * E';
end
Wrot    = ops.scaleproc * Wrot;

fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fid         = fopen(ops.fbinary, 'r');
if ~ops.useRAM
    fidW        = fopen(ops.fproc,   'w');
end

%
for ibatch = 1:Nbatch
    offset = max(0, twind + 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    if offset==0
        ioffset = 0;
    else
        ioffset = ops.ntbuff;
    end
    fseek(fid, offset, 'bof');
    
    buff = fread(fid, [NchanTOT NTbuff], '*int16');
    if isempty(buff)
        break;
    end
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end
    
    if ops.GPU
        dataRAW = gpuArray(buff);
    else
        dataRAW = buff;
    end
    dataRAW = dataRAW';
    dataRAW = single(dataRAW);
    dataRAW = dataRAW(:, chanMap);
    
    % CAR, common average referencing by median
    dataRAW = dataRAW - mean(dataRAW, 1);
    
    datr = filter(b1, a1, dataRAW);
    datr = flipud(datr);
    datr = filter(b1, a1, datr);
    datr = flipud(datr);
    
    % CAR, common average referencing by median
    if getOr(ops, 'CAR', 1)
        datr = datr - median(datr, 2);
    end
    
    datr = datr(ioffset + (1:NT),:);
    
    datr    = datr * Wrot;
    
    if ops.useRAM
        DATA(:,:,ibatch) = gather_try(datr);
    else
        datcpu  = gather_try(int16(datr));
        fwrite(fidW, datcpu, 'int16');
    end
end

Wrot        = gather_try(Wrot);
rez.Wrot    = Wrot;

fclose(fidW);
fclose(fid);


rez.temp.Nbatch = Nbatch;

