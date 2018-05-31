ops.chanMap             = 'D:\GitHub\KiloSort2\configFiles\neuropixPhase3A_kilosortChanMap.mat';
% ops.chanMap = 1:ops.Nchan; % treated as linear probe if a chanMap file

ops.fs                  = 30000;        % sample rate

ops.fshigh              = 150;   % frequency for high pass filtering

ops.Th       = 10;     % threshold on projections (like in Kilosort1, can be different for last pass like [10 8])

ops.lam      = 40^2;   % weighting on the amplitude penalty (like in Kilosort1, but it has to be much larger)

ops.ThS      = [8 8];  % lower bound on acceptable single spike quality (annealed)

ops.momentum = [20 400]; % number of samples to average over (annealed) 

ops.minFR    = 1/50; % minimum spike rate (Hz)

ops.sigmaMask  = 30; % spatial constant in um for computing residual variance of spike

ops.Nfilt       = 1024; % max number of clusters

ops.nfullpasses = 1; % how many forward backward passes to do

ops.nPCs        = 3; % how many PCs to project the spikes into

ops.ccsplit = 0.99; % required isolation for splitting a cluster at the end (max = 1)

ops.useRAM = 0; % whether to hold data in RAM (won't check if there's enough RAM)

ops.ThPre = 8; % threshold crossings for pre-clustering (in PCA projection space)


%% danger, changing these settings can lead to fatal errors
ops.GPU                 = 1; % whether to run this code on an Nvidia GPU (much faster, mexGPUall first)

ops.nSkipCov            = 5; % compute whitening matrix from every N-th batch (1)

ops.ntbuff              = 64;    % samples of symmetrical buffer for whitening and spike detection

ops.scaleproc           = 200;   % int16 scaling of whitened data

ops.NT                  = 64*1024+ ops.ntbuff;% this is the batch size (try decreasing if out of memory) 
% for GPU should be multiple of 32 + ntbuff

% options for determining PCs
ops.spkTh           = -6;      % spike threshold in standard deviations (-6)

ops.loc_range       = [5  4];  % ranges to detect peaks; plus/minus in time and channel ([3 1])
ops.long_range      = [30  6]; % ranges to detect isolated peaks ([30 6])
ops.maskMaxChannels = 5;       % how many channels to mask up/down ([5])

ops.criterionNoiseChannels = 0.2; % fraction of "noise" templates allowed to span all channel groups (see createChannelMapFile for more info). 

ops.whiteningRange = 32;
%%