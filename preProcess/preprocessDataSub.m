function rez = preprocessDataSub(ops)
% this function takes an ops struct, which contains all the Kilosort2 settings and file paths
% and creates a new binary file of preprocessed data, logging new variables into rez.
% The following steps are applied:
% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

tic;
ops.nt0 	  = getOr(ops, {'nt0'}, 61); % number of time samples for the templates (has to be <=81 due to GPU shared memory)
ops.nt0min  = getOr(ops, 'nt0min', ceil(20 * ops.nt0/61)); % time sample where the negative peak should be aligned

NT       = ops.NT ; % number of timepoints per batch
NchanTOT = ops.NchanTOT; % total number of channels in the raw binary file, including dead, auxiliary etc

bytes       = get_file_size(ops.fbinary); % size in bytes of raw binary
nTimepoints = floor(bytes/NchanTOT/2); % number of total timepoints
ops.tstart  = ceil(ops.trange(1) * ops.fs); % starting timepoint for processing data segment
ops.tend    = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); % ending timepoint
ops.sampsToRead = ops.tend-ops.tstart; % total number of samples to read
ops.twind = ops.tstart * NchanTOT*2; % skip this many bytes at the start

Nbatch      = ceil(ops.sampsToRead /NT); % number of data batches
ops.Nbatch = Nbatch;

[chanMap, xc, yc, kcoords, NchanTOTdefault] = loadChanMap(ops.chanMap); % function to load channel map file
ops.NchanTOT = getOr(ops, 'NchanTOT', NchanTOTdefault); % if NchanTOT was left empty, then overwrite with the default

ops.igood = true(size(chanMap));

ops.Nchan = numel(chanMap); % total number of good channels that we will spike sort
ops.Nfilt = getOr(ops, 'nfilt_factor', 4) * ops.Nchan; % upper bound on the number of templates we can have

rez.ops         = ops; % memorize ops

rez.xc = xc; % for historical reasons, make all these copies of the channel coordinates
rez.yc = yc;
rez.xcoords = xc;
rez.ycoords = yc;
% rez.connected   = connected;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords;


NTbuff      = NT + 3*ops.ntbuff; % we need buffers on both sides for filtering

rez.ops.Nbatch = Nbatch;
rez.ops.NTbuff = NTbuff;
rez.ops.chanMap = chanMap;


fprintf('Time %3.0fs. Computing whitening matrix.. \n', toc);

% this requires removing bad channels first
Wrot = get_whitening_matrix(rez); % outputs a rotation matrix (Nchan by Nchan) which whitens the zero-timelag covariance of the data
% Wrot = gpuArray.eye(size(Wrot,1), 'single');
% Wrot = diag(Wrot);

fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fid         = fopen(ops.fbinary, 'r'); % open for reading raw data
if fid<3
    error('Could not open %s for reading.',ops.fbinary);
end
fidW        = fopen(ops.fproc,   'w+'); % open for writing processed data
if fidW<3
    error('Could not open %s for writing.',ops.fproc);    
end

% weights to combine batches at the edge
w_edge = linspace(0, 1, ops.ntbuff)';
ntb = ops.ntbuff;
datr_prev = gpuArray.zeros(ntb, ops.Nchan, 'single');

for ibatch = 1:Nbatch
    % we'll create a binary file of batches of NT samples, which overlap consecutively on ops.ntbuff samples
    % in addition to that, we'll read another ops.ntbuff samples from before and after, to have as buffers for filtering
    offset = max(0, ops.twind + 2*NchanTOT*(NT * (ibatch-1) - ntb)); % number of samples to start reading at.
    
    fseek(fid, offset, 'bof'); % fseek to batch start in raw file

    buff = fread(fid, [NchanTOT NTbuff], '*int16'); % read and reshape. Assumes int16 data (which should perhaps change to an option)
    if isempty(buff)
        break; % this shouldn't really happen, unless we counted data batches wrong
    end
    nsampcurr = size(buff,2); % how many time samples the current batch has
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr); % pad with zeros, if this is the last batch
    end
    if offset==0
        bpad = repmat(buff(:,1), 1, ntb);
        buff = cat(2, bpad, buff(:, 1:NTbuff-ntb)); % The very first batch has no pre-buffer, and has to be treated separately
    end
    
    datr    = gpufilter(buff, ops, chanMap); % apply filters and median subtraction
    
%     datr(ntb + [1:ntb], :) = datr_prev;
    datr(ntb + [1:ntb], :) = w_edge .* datr(ntb + [1:ntb], :) +...
        (1 - w_edge) .* datr_prev;
   
    datr_prev = datr(ntb +NT + [1:ops.ntbuff], :);
    datr    = datr(ntb + (1:NT),:); % remove timepoints used as buffers
   
    datr    = datr * Wrot; % whiten the data and scale by 200 for int16 range

    datcpu  = gather(int16(datr')); % convert to int16, and gather on the CPU side
    count = fwrite(fidW, datcpu, 'int16'); % write this batch to binary file
    if count~=numel(datcpu)
        error('Error writing batch %g to %s. Check available disk space.',ibatch,ops.fproc);
    end
end
fclose(fidW); % close the files
fclose(fid);

rez.Wrot    = gather(Wrot); % gather the whitening matrix as a CPU variable

fprintf('Time %3.0fs. Finished preprocessing %d batches. \n', toc, Nbatch);

rez.temp.Nbatch = Nbatch;
