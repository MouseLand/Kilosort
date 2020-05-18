function igood = get_good_channels(ops, chanMap)
% of the channels indicated by the user as good (chanMap)
% further subset those that have a mean firing rate above a certain value
% (default is ops.minfr_goodchannels = 0.1Hz)
% needs the same filtering parameters in ops as usual
% also needs to know where to start processing batches (twind)
% and how many channels there are in total (NchanTOT)

Nbatch = ops.Nbatch;
twind = ops.twind;
NchanTOT = ops.NchanTOT;
NT = ops.NT;
Nchan = numel(chanMap);

% load data into patches, filter, compute covariance
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
end

fid = fopen(ops.fbinary, 'r');
% irange = [NT/8:(NT-NT/8)];

ibatch = 1;
ich = gpuArray.zeros(5e4,1, 'int16');
k = 0;
ttime = 0;

% from a subset of batches, count threshold crossings
while ibatch<=Nbatch
    offset = twind + 2*NchanTOT*NT* (ibatch-1);
    fseek(fid, offset, 'bof');
    buff = fread(fid, [NchanTOT NT], '*int16');

    if isempty(buff)
        break;
    end

    datr    = gpufilter(buff, ops, chanMap); % apply filters and median subtraction

    % very basic threshold crossings calculation
    datr = datr./std(datr,1,1); % standardize each channel ( but don't whiten)

    mdat = my_min(datr, 30, 1); % get local minima as min value in +/- 30-sample range
    ind = find(datr<mdat+1e-3 & datr<ops.spkTh); % take local minima that cross the negative threshold
    [xi, xj] = ind2sub(size(datr), ind); % back to two-dimensional indexing
    xj(xi<ops.nt0 | xi>NT-ops.nt0) = []; % filtering may create transients at beginning or end. Remove those.
    if k+numel(xj)>numel(ich)
        ich(2*numel(ich)) = 0; % if necessary, extend the variable which holds the spikes
    end
    ich(k + [1:numel(xj)]) = xj; % collect the channel identities for the detected spikes

    k = k + numel(xj);

    ibatch = ibatch + ceil(Nbatch/100); % skip every 100 batches
    ttime = ttime + size(datr,1)/ops.fs; % keep track of total time where we took spikes from
end
fclose(fid);

ich = ich(1:k);

nc = histcounts(ich, .5 + [0:Nchan]); % count how many spikes each channel got
nc = nc/ttime; % divide by total time to get firing rate

% igood = nc>.1;
igood = nc>=getOr(ops, 'minfr_goodchannels', .1); % keep only those channels above the preset mean firing rate

fprintf('found %d threshold crossings in %2.2f seconds of data \n', k, ttime)
fprintf('found %d bad channels \n', sum(~igood))
