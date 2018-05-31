addpath(genpath('D:\GitHub\KiloSort2')) % path to kilosort folder
addpath('D:\GitHub\npy-matlab')

pathToYourConfigFile = 'D:\GitHub\KiloSort2\configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFile384.m'))

% common options for every probe
ops.NchanTOT    = 385; % total number of channels in your recording
ops.trange      = [3750 Inf]; % TIME RANGE IN SECONDS TO PROCESS

% find the binary file in this folder
rootZ = 'H:\DATA\Spikes\Robbins\ZNP1';
fs = dir(fullfile(rootZ, '*.bin'));
fname = fs(1).name;
ops.fbinary     = fullfile(rootZ,  fname);

% path to whitened, filtered proc file (on a fast SSD)
rootH = 'H:\DATA\Spikes\temp\';
ops.fproc       = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD

%%
% preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);

% pre-clustering to re-order batches by depth
rez = clusterSingleBatches(rez);

%% main optimization
rez = learnAndSolve8(rez);

% this does splits
rez    = splitAllClusters(rez);

%% this saves to Phy
rezToPhy(rez, rootZ);

% fname = fullfile(rootZ, 'rez.mat');
% save(fname, 'rez', '-v7.3');


