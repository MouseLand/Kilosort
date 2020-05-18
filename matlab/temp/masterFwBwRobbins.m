addpath(genpath('D:\GitHub\KiloSort2')) % path to kilosort folder
addpath('D:\GitHub\npy-matlab')

pathToYourConfigFile = 'D:\GitHub\KiloSort2\configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFileBench384.m'))

% common options for every probe
ops.chanMap     = 'D:\GitHub\KiloSort2\configFiles\neuropixPhase3A_kilosortChanMap.mat';
ops.trange      = [0 Inf]; % TIME RANGE IN SECONDS TO PROCESS

 % these settings overwrite any settings from config
ops.Th       = [10 10];     % threshold on projections (like in Kilosort1)
ops.lam      = 40^2;   % weighting on the amplitude penalty (like in Kilosort1, but it has to be much larger)

ops.ThS      = [8 8];  % lower bound on acceptable single spike quality
ops.momentum = [20 400]; % number of samples to average over
ops.minFR    = 1/50; % minimum spike rate (Hz)

ops.sigmaMask  = 30;

ops.Nfilt       = 1024; % max number of clusters
ops.nfullpasses = 3; % how many forward backward passes to do
ops.nPCs        = 3; % how many PCs to project the spikes into

ops.NchanTOT = 385;
ops.useRAM = 0;
ops.ccsplit = .99;

% rootZ = 'D:\DATA\ALLEN\mouse366119\probeC_2018-03-02_15-18-32_SN619041624\experiment1\recording1\continuous\Neuropix-120.0\';
% rootZ = 'H:\DATA\Spikes\Robbins\ZNP1';
rootZ = 'H:\DATA\Spikes\WillAllen';
rootH = 'H:\DATA\Spikes\temp\';

fs = dir(fullfile(rootZ, '*.bin'));

fname = fs(1).name;

ops.fbinary     = fullfile(rootZ,  fname);
ops.fproc       = fullfile(rootH, 'temp_wh2.dat'); % residual from RAM of preprocessed data


% preprocess data
rez = preprocessDataSub(ops);

rez = clusterSingleBatches(rez);

% figure(191);
% imagesc(rez.ccb(rez.iorig, rez.iorig), [20 100])
%%
learnAndSolve8
% rez = learnAndSolve8(rez);

rez2    = splitAllClusters(rez);

rezToPhy(rez2, rootZ);

