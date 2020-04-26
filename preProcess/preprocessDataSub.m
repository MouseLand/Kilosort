function [rez, DATA] = preprocessDataSub(ops)
tic;
ops.nt0 	= getOr(ops, {'nt0'}, 61);
ops.nt0min  = getOr(ops, 'nt0min', ceil(20 * ops.nt0/61));

NT       = ops.NT ;
NchanTOT = ops.NchanTOT;

bytes = get_file_size(ops.fbinary);
nTimepoints = floor(bytes/NchanTOT/2);
ops.tstart = ceil(ops.trange(1) * ops.fs);
ops.tend   = min(nTimepoints, ceil(ops.trange(2) * ops.fs));
ops.sampsToRead = ops.tend-ops.tstart;
ops.twind = ops.tstart * NchanTOT*2;

Nbatch      = ceil(ops.sampsToRead /(NT-ops.ntbuff));
ops.Nbatch = Nbatch;

[chanMap, xc, yc, kcoords, NchanTOTdefault] = loadChanMap(ops.chanMap);
ops.NchanTOT = getOr(ops, 'NchanTOT', NchanTOTdefault);

if getOr(ops, 'minfr_goodchannels', .1)>0
    
    % determine bad channels
    fprintf('Time %3.0fs. Determining good channels.. \n', toc);

    igood = get_good_channels(ops, chanMap);
    xc = xc(igood);
    yc = yc(igood);
    kcoords = kcoords(igood);
    chanMap = chanMap(igood);
        
    ops.igood = igood;
else
    ops.igood = true(size(chanMap));
end

ops.Nchan = numel(chanMap);
ops.Nfilt = getOr(ops, 'nfilt_factor', 4) * ops.Nchan;

rez.ops         = ops;
rez.xc = xc;
rez.yc = yc;

rez.xcoords = xc;
rez.ycoords = yc;

% rez.connected   = connected;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords; 


NTbuff      = NT + 4*ops.ntbuff;


% by how many bytes to offset all the batches
rez.ops.Nbatch = Nbatch;
rez.ops.NTbuff = NTbuff;
rez.ops.chanMap = chanMap;


fprintf('Time %3.0fs. Computing whitening matrix.. \n', toc);

% this requires removing bad channels first
Wrot = get_whitening_matrix(rez);


fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fid         = fopen(ops.fbinary, 'r');
if ~ops.useRAM
    fidW        = fopen(ops.fproc,   'w');
    DATA = [];
else
    DATA = zeros(NT, rez.ops.Nchan, Nbatch, 'int16');    
end
% load data into patches, filter, compute covariance
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
end

if isfield(ops, 'driftT'); 
    fprintf('applying drift correction!\n'); 
    
    fidShift        = fopen(fullfile(fileparts(ops.fproc), 'shifted.bin'),   'w');
    
    method = 'v5cubic'; % 'spline' not working on gpu?
    nInterp = 205; 
    ux = unique(xc);
    for sg = 1:numel(ux)
                
        thisGroup = xc==ux(sg);
        ycoords = yc(thisGroup);
        gridSize = median(diff(ycoords)); % should be 20um for imec probes

        numZeros{sg} = ceil(max(abs(ops.driftPos))/gridSize)+2;

        % technically you only need to pad one side of the matrix, on the trailing
        % end of the shift; this isn't implemented here but would save some time
        % and memory
        paddedX{sg} = single(gpuArray([ (-numZeros{sg}:1:-1)*gridSize+ycoords(1) ycoords' (1:numZeros{sg})*gridSize+ycoords(end) ]));        
        paddedData{sg} = single(gpuArray(zeros(numel(ycoords)+2*numZeros{sg}, NT)));
    end
    
end

for ibatch = 1:Nbatch
    if mod(ibatch, 20)==1
        fprintf(1, '%d/%d...\n', ibatch, Nbatch); 
    end
    offset = max(0, ops.twind + 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
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
    
    % subtract the mean from each channel
    dataRAW = dataRAW - mean(dataRAW, 1);   
    
    % CAR, common average referencing by median
    if getOr(ops, 'CAR', 1)
        dataRAW = dataRAW - median(dataRAW, 2);
    end
    
    datr = filter(b1, a1, dataRAW);
    datr = flipud(datr);
    datr = filter(b1, a1, datr);
    datr = flipud(datr);
    
    datr = datr(ioffset + (1:NT),:);
    
    datr    = datr * Wrot;
    
    if isfield(ops, 'driftT') && isfield(ops, 'driftPos')
        % determine the position at this batch (mean of position at all samples
        % of the batch)
        batchStartSamp = offset/2/NchanTOT;
        batchPos = nanmean(interp1(ops.driftT, ops.driftPos, (batchStartSamp:(batchStartSamp+NTbuff))/ops.fs));
        if batchPos~=0
            
            for sg = 1:numel(ux)
                
                thisGroup = xc==ux(sg);
                
                % shift aligned sets of chans by defined amount
                %shiftedData = datShift(gather_try(datr(:,thisGroup)'), yc(thisGroup), batchPos);
                %datr(:,thisGroup) = gpuArray(shiftedData'); % re-assemble data
                
                thisData = datr(:,thisGroup)';
                paddedData{sg}(numZeros{sg}+1:end-numZeros{sg},:) = thisData;
                px = paddedX{sg}'; pd = paddedData{sg}; newx = paddedX{sg}+gpuArray(batchPos);
                shiftedPaddedData = zeros(size(pd), 'like', pd);
                for q = 1:NT/nInterp
                    qidx = (q-1)*nInterp+1:q*nInterp;
                    shiftedPaddedData(:,qidx) = interp1(px,pd(:,qidx),newx,method);
                end
                shiftedData = shiftedPaddedData(numZeros{sg}+1:end-numZeros{sg},:);
                datr(:,thisGroup) = shiftedData';
               
                
            end
        end

        if ibatch==1
            datOut = gather_try(int16(datr(1:(end-2*ops.ntbuff),:)'));
        else
            datOut = gather_try(int16(datr(1:(end-ops.ntbuff),:)')); 
        end
        fwrite(fidShift, datOut, 'int16'); 
    end
    
    
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
if isfield(ops, 'driftT'); fclose(fidShift); end

fprintf('Time %3.0fs. Finished preprocessing %d batches. \n', toc, Nbatch);

rez.temp.Nbatch = Nbatch;

