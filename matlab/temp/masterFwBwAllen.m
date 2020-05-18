addpath(genpath('D:\GitHub\KiloSort2')) % path to kilosort folder
addpath('D:\GitHub\npy-matlab')

pathToYourConfigFile = 'D:\GitHub\KiloSort2\configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFile384.m'))

ops.chanMap  = 'D:\GitHub\KiloSort2\configFiles\neuropixPhase3B2_kilosortChanMap.mat';

% common options for every probe
ops.NchanTOT    = 384;% total number of channels in your recording
ops.trange      =[0 Inf];% TIME RANGE IN SECONDS TO PROCESS

% find the binary file in this folder
rootZ = 'H:\DATA\Spikes\Allen\recording1';
fs = [dir(fullfile(rootZ, '*.dat')) dir(fullfile(rootZ, '*.bin'))];
fname = fs(1).name;
ops.fbinary     = fullfile(rootZ,  fname);

% path to whitened, filtered proc file (on a fast SSD)
rootH = 'H:\DATA\Spikes\temp\';
ops.fproc       = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD


% preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);

%%
% pre-clustering to re-order batches by depth
% rez = clusterSingleBatches(rez);
clusterSingleBatches2;

%%
% main optimization
rez = learnAndSolve8(rez);

% this does splits
rez    = splitAllClusters(rez);

% this saves to Phy
rezToPhy(rez, rootZ);

fname = fullfile(rootZ, 'rez.mat');
save(fname, 'rez', '-v7.3');


