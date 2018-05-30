function [rez, DATA] = preprocessDataSub(ops)
tic;
ops.nt0 	= getOr(ops, {'nt0'}, 61);

if ~isempty(ops.chanMap)
    if ischar(ops.chanMap)
        load(ops.chanMap);
        try
            chanMapConn = chanMap(connected>1e-6);
            xc = xcoords(connected>1e-6);
            yc = ycoords(connected>1e-6);
        catch
            chanMapConn = 1+chanNums(connected>1e-6);
            xc = zeros(numel(chanMapConn), 1);
            yc = [1:1:numel(chanMapConn)]';
        end
        ops.Nchan    = getOr(ops, 'Nchan', sum(connected>1e-6));
        ops.NchanTOT = getOr(ops, 'NchanTOT', numel(connected));
        if exist('fs', 'var')
            ops.fs       = getOr(ops, 'fs', fs);
        end
    else
        chanMap = ops.chanMap;
        chanMapConn = ops.chanMap;
        xc = zeros(numel(chanMapConn), 1);
        yc = [1:1:numel(chanMapConn)]';
        connected = true(numel(chanMap), 1);      
        
        ops.Nchan    = numel(connected);
        ops.NchanTOT = numel(connected);
    end
else
    chanMap  = 1:ops.Nchan;
    connected = true(numel(chanMap), 1);
    
    chanMapConn = 1:ops.Nchan;    
    xc = zeros(numel(chanMapConn), 1);
    yc = [1:1:numel(chanMapConn)]';
end
if exist('kcoords', 'var')
    kcoords = kcoords(connected);
else
    kcoords = ones(ops.Nchan, 1);
end
NchanTOT = ops.NchanTOT;
NT       = ops.NT ;

rez.ops         = ops;
rez.xc = xc;
rez.yc = yc;
if exist('xcoords', 'var')
   rez.xcoords = xcoords;
   rez.ycoords = ycoords;
else
   rez.xcoords = xc;
   rez.ycoords = yc;
end
rez.connected   = connected;
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
    dataRAW = dataRAW(:, chanMapConn);
    
    datr = filter(b1, a1, dataRAW);
    datr = flipud(datr);
    datr = filter(b1, a1, datr);
    datr = flipud(datr);

    % common average referencing
    datr = datr - median(datr, 2);
    
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
    %
    [E, D] 	= svd(CC);
    D = diag(D);
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
    dataRAW = dataRAW(:, chanMapConn);
    
    datr = filter(b1, a1, dataRAW);
    datr = flipud(datr);
    datr = filter(b1, a1, datr);
    datr = flipud(datr);
     % common average referencing
    datr = datr - median(datr, 2);
    
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
if ops.verbose
    fprintf('Time %3.2f. Whitened data written to disk... \n', toc);
    fprintf('Time %3.2f. Preprocessing complete!\n', toc);
end


rez.temp.Nbatch = Nbatch;

