function make_noise_model
%% Input params about your paths and data
%start by building a rez structure -- will be used to call the whitening
%and filtering functions in KS2

%these need to match the current location of KS2 and npy

addpath(genpath('D:\KS2\Kilosort2-master')) % path to kilosort folder
addpath('D:\KS2\npy-matlab-master\')

% params normally read in from a config file. These need to match the model
% data set from which the noise will be extracted.

% sample rate
ops.fs = 30000;  
ops.bitPerUV = 0.42667;

% chanMap for your input data set. Make sure it includes reference channels
% where they occur in the data
ops.chanMap = 'D:\SC026_080819_g0_Imec2\SC026_noise_model\neuropixPhase3B1_kilosortChanMap.mat';
[~,chanMapName,~] = fileparts(ops.chanMap);

%path to your data file; results will also be written to this folder
rootZ = 'D:\SC026_080819_g0_Imec2\SC026_noise_model\';
%name of the binary file
dataName = 'SC026_080819_g0_tcat.imec2.ap.bin';

%directory for temporary whitened data filt
rootH = 'D:\KS2\kilosort_datatemp\';
ops.trange = [0 inf]; % time range to use when extracting the whitening matrix
ops.NchanTOT    = 385; % total number of channels in your recording, including digital
%Time to sample in sec, need to sample at least enough to freq down to the high pass limit
clipDur = 0.25; 
%%

% frequency for high pass filtering (150)
ops.fshigh = 150;  

%processing params usually read in from config. These should not be changed
ops.GPU                 = 1; % has to be 1, no CPU version yet, sorry
ops.useRAM              = 0;
ops.ntbuff              = 64;    % samples of symmetrical buffer for whitening and spike detection
ops.NT                  = 64*1024+ ops.ntbuff; % must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory). 
ops.whiteningRange      = 32; % number of channels to use for whitening each channel
ops.nSkipCov            = 25; % compute whitening matrix from every N-th batch
ops.scaleproc           = 100;   % int16 scaling of whitened data


ops.fproc       = fullfile(rootH, 'temp_wh.imec.ap.bin'); % proc file on a fast SSD

ops.fbinary = fullfile(rootZ, dataName);
ops.minfr_goodchannels = 0; %don't remove any channels due to low firing rate

% call preprocessDataSub to filter data and create the whitening matrix.
ops.trange = [0 60]; % time range to use when extracting the whitening matrix
[rez, ops] = preprocessDataForNoise(ops);

%calculate fft for each channel in the whitened data (stored in ops.fproc)
tic;
[fftSamp, nT_fft] = sampFFT( rez, clipDur );
fprintf( 'time to calc fft: %f\n', toc );
%use the fft to generate noise for each channel for a time period 
%Can only make batches of nT points at a time

    noiseT = 2.2;           %in seconds, chosen for ~66000 points
    noiseSamp = noiseT*rez.ops.fs;
    nChan = numel(rez.ops.chanMap);
    eNoise = zeros( noiseSamp, nChan, 'single' );
    
    noiseBatch = ceil(noiseSamp/nT_fft);    
    lastWind = noiseSamp - (noiseBatch-1)*nT_fft; %in samples
    fprintf( 'noiseSamp, batchSamp, lastWind: %d, %d, %d\n', noiseSamp, nT_fft, lastWind);
    
    for j = 1:nChan     
            for i = 1:noiseBatch-1
                tStart = (i-1)*nT_fft+1;
                tEnd = i * nT_fft;            
                eNoise(tStart:tEnd,j) = fftnoise(fftSamp(:,j),1);
            end
            %for last batch, call one more time and truncate
            tempNoise = fftnoise(fftSamp(:,j),1);
            tStart = (noiseBatch-1)*nT_fft+1;
            tEnd = noiseSamp;
            eNoise(tStart:tEnd,j) = tempNoise(1:lastWind);             
    end
    
    selectChan = [10,11,12,13,14];  
        
    tR = [4200:5200]; %just looking at 1000 samples
    h = figure(1);  
    for k = 1:numel(selectChan)  
        currNoise = eNoise(tR,selectChan(k));
        plot( tR, currNoise + k*500 );
        hold on;
    end  
    title("1000 samples of generated noise, before unwhitening");
    hold off

    
    %unwhiten this noise array
    eNoise_unwh = eNoise/rez.Wrot;
    
    %to get the final noise array, map to an array including all channels
    nAllChan = max(rez.ops.chanMap);
    eNoise_final = zeros(noiseSamp, nAllChan, 'single');
    eNoise_final(:,rez.ops.chanMap) = eNoise_unwh;
    

    tR = [4200:5200]; %plot 1000 samples post unwhitening
    h = figure(2);  
    for k = 1:numel(selectChan)  
        currNoise = eNoise_final(tR,selectChan(k));
        plot( tR, currNoise + k*50 );
        hold on;
    end   
    title('1000 samples of generated noise, after unwhitening');
    hold off

    nm.chanMapName = chanMapName;
    nm.fft = fftSamp;
    nm.nt = nT_fft;
    nm.Wrot = rez.Wrot;
    nm.chanMap = rez.ops.chanMap;
    nm.bitPerUV =  rez.ops.bitPerUV;
    
%save rez file
fprintf('Saving rez file  \n');
fname = fullfile(rootZ, 'rez.mat');
save(fname, 'rez', '-v7.3');

%save noise model file
fprintf('Saving nm file  \n');
fname = fullfile(rootZ, 'noiseModel.mat');
save(fname, 'nm', '-v7.3');

end



function [rez, ops] = preprocessDataForNoise(ops)

%build rez structure -- code taken from preprocessDataSub
%Only changes are to skip checking for low spike rate samples and
%write out the data as rows = time -- just a convenience so it 
%it can be read in using the same code

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
ops.igood = true(size(chanMap));

ops.Nchan = numel(chanMap);
ops.Nfilt = getOr(ops, 'nfilt_factor', 4) * ops.Nchan;

rez.ops         = ops;
rez.xc = xc;
rez.yc = yc;

rez.xcoords = xc;
rez.ycoords = yc;

rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords; 


NTbuff      = NT + 4*ops.ntbuff;

% by how many bytes to offset all the batches
rez.ops.Nbatch = Nbatch;
rez.ops.NTbuff = NTbuff;
rez.ops.chanMap = chanMap;

%build whitening matrix. This function filters before calculating the
%cross correlation

tic;

fprintf('Time %3.0fs. Computing whitening matrix.. \n', toc);

Wrot = get_whitening_matrix(rez);

%Apply the whitening matrix to a subset of the original data

ops.tstart = ceil(ops.trange(1) * ops.fs);
ops.tend   = min(nTimepoints, ceil(ops.trange(2) * ops.fs));
ops.sampsToRead = ops.tend-ops.tstart;
ops.twind = ops.tstart * NchanTOT*2;
Nbatch      = ceil(ops.sampsToRead /(NT-ops.ntbuff));
ops.Nbatch = Nbatch;

fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fid         = fopen(ops.fbinary, 'r');
if ~ops.useRAM
    fidW        = fopen(ops.fproc,   'w');
    DATA = [];
else
    DATA = zeros(NT, rez.ops.Nchan, Nbatch, 'int16');    
end
% load data into batches, filter, compute covariance
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
elseif isfield(ops,'fshigh')
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
end

for ibatch = 1:Nbatch
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
    
    if (isfield(ops,'fshigh'))
        datr = filter(b1, a1, dataRAW);
        datr = flipud(datr);
        datr = filter(b1, a1, datr);
        datr = flipud(datr);
    else
        datr = dataRAW;
    end
    
    % CAR, common average referencing by median
    if getOr(ops, 'CAR', 1)
        datr = datr - median(datr, 2);
    end
    
    datr = datr(ioffset + (1:NT),:);
    
    %apply whitening
    datr    = datr * Wrot;
    
    if ops.useRAM
        DATA(:,:,ibatch) = gather_try(datr);
    else
        datcpu  = gather_try(int16(datr));
        %next step will be reading in this file to get FTs
        %Most convenient for that process to have the data in the original
        %format (row = time, column = channel index)
        fwrite(fidW, datcpu', 'int16');
    end
end

Wrot        = gather_try(Wrot);
rez.Wrot    = Wrot;

fclose(fidW);
fclose(fid);


end


function [fftSamp, nT] = sampFFT( rez, clipDur )

    ops = rez.ops;
    
    %get file of whitened data
    datFile = ops.fproc;
    
    %read in some data
    nChansInFile = numel(ops.chanMap);  % channels in whitenened data, excludes ref chans in original 
    d = dir(ops.fproc); 
    nSamps = d.bytes/2/nChansInFile;
    tSkip = 1.0; % skip first tSkip seconds
    
    sampStart = round(ops.fs*tSkip); 
    nClipSamps = round(ops.fs*clipDur);
    mmf = memmapfile(datFile, 'Format', {'int16', [nChansInFile nSamps], 'x'});
    thisDat = (double(mmf.Data.x(:, (1:nClipSamps)+sampStart)));
    %subtract the DC
    thisDat = bsxfun(@minus, thisDat, mean(thisDat,2));
    [~,nT] = size(thisDat);
    %data is formatted as Nchan rows by nt columns. fft returns the
    %transform of the columns, so transpose
    fftSamp = fft(thisDat');
    
end
 

function noise=fftnoise(f,Nseries)
% Generate noise with a given power spectrum.
% Useful helper function for Monte Carlo null-hypothesis tests and confidence interval estimation.
%  
% noise=fftnoise(f[,Nseries])
%
% INPUTS:
% f: the fft of a time series (must be a column vector)
% Nseries: number of noise series to generate. (default=1)
% 
% OUTPUT:
% noise: surrogate series with same power spectrum as f. (each column is a surrogate).
%
%   --- Aslak Grinsted (2009)
if nargin<2
    Nseries=1;
end
f=f(:);     %ensures f is a column vector
N=length(f); 
Np=floor((N-1)/2);
phases=rand(Np,Nseries)*2*pi;
phases=complex(cos(phases),sin(phases)); % this was the fastest alternative in my tests. 
f=repmat(f,1,Nseries);
f(2:Np+1,:)=f(2:Np+1,:).*phases;
f(end:-1:end-Np+1,:)=conj(f(2:Np+1,:));
noise=real(ifft(f,[],1)); 

end

