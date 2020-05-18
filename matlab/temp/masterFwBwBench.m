addpath(genpath('D:\GitHub\KiloSort2')) % path to kilosort folder
addpath('D:\GitHub\npy-matlab')

pathToYourConfigFile = 'D:\GitHub\KiloSort2\configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFileBench384.m'))

% common options for every probe
ops.chanMap = 'D:\DATA\Neuropixels\forPRBimecToWhisper.mat';
ops.trange      = [0 Inf]; % TIME RANGE IN SECONDS TO PROCESS

 % these settings overwrite any settings from config
ops.Th       = 12;     % threshold on projections (like in Kilosort1)
ops.lam      = 40^2;   % weighting on the amplitude penalty (like in Kilosort1, but it has to be much larger)

ops.ThS      = [10 15];  % lower bound on acceptable single spike quality
ops.momentum = [20 400]; % number of samples to average over
ops.minFR    = 1/50; % minimum spike rate (Hz)

ops.sigmaMask  = 30;

ops.Nfilt       = 512; % max number of clusters
ops.nfullpasses = 3; % how many forward backward passes to do
ops.nPCs        = 3; % how many PCs to project the spikes into

ops.useRAM = 0;


rootZ = '';

rootH = 'H:\DATA\Spikes\temp\';
fname = 'continuous.dat';

ops.fbinary     = fullfile(rootZ,  fname);
ops.fproc       = fullfile(rootH, 'temp_wh.dat'); % residual from RAM of preprocessed data


% preprocess data
rez = preprocessDataSub(ops);

fname = fullfile(rootZ, 'rez.mat');
save(fname, 'rez');

% cluster the threshold crossings
learnAndSolve7;

rezToPhy(rez, rootZ);

