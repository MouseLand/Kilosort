addpath(genpath('D:\GitHub\KiloSort2')) % path to kilosort folder
addpath('D:\GitHub\npy-matlab')

pathToYourConfigFile = 'D:\GitHub\KiloSort2\configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFileBench384.m'))

% common options for every probe
ops.chanMap     = 'D:\GitHub\KiloSort2\configFiles\neuropixPhase3A_kilosortChanMap_385.mat';
ops.trange      = [3750 Inf]; % TIME RANGE IN SECONDS TO PROCESS

 % these settings overwrite any settings from config
ops.Th       = 10;     % threshold on projections (like in Kilosort1)
ops.lam      = 40^2;   % weighting on the amplitude penalty (like in Kilosort1, but it has to be much larger)

ops.ThS      = [10 15];  % lower bound on acceptable single spike quality
ops.momentum = [20 400]; % number of samples to average over
ops.minFR    = 1/50; % minimum spike rate (Hz)

ops.sigmaMask  = 30;

ops.Nfilt       = 512; % max number of clusters
ops.nfullpasses = 3; % how many forward backward passes to do
ops.nPCs        = 3; % how many PCs to project the spikes into

ops.useRAM = 0;

probeName = {'K1', 'K2', 'K3', 'ZNP1', 'ZNP2', 'ZNP3', 'ZNP4', 'ZO'};
mname = 'Robbins'; %'Waksman'; %'Krebs'; %'Robbins';
datexp = '2017-06-13'; %'2017-06-10'; %'2017-06-13';
rootZ = 'H:\DATA\Spikes';

% rez.ops.minFR = 1/50;

tic
%%

for j = 4
    fname = sprintf('%s_%s_%s_g0_t0.imec.ap_CAR.bin', mname, datexp, probeName{j});
    ops.fbinary     = fullfile(rootZ,  mname, fname);    
    ops.fproc       = 'H:\DATA\Spikes\temp_wh.dat'; % residual from RAM of preprocessed data
    ops.dir_rez     = 'H:\DATA\Spikes\';
    
    % preprocess data
    rez = preprocessDataSub(ops);    
   %%
    fname = fullfile(ops.dir_rez,  ...
        sprintf('rez_%s_%s_%s.mat', mname, datexp, probeName{j}));
    save(fname, 'rez');
    
    % cluster the threshold crossings
    learnAndSolve7;
    %%
    savePath = fullfile('H:\DATA\Spikes\', mname);    
    rezToPhy(rez, savePath);

    % save the result file
end
