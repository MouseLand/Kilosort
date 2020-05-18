addpath(genpath('D:\GitHub\KiloSort2')) % path to kilosort folder
addpath('D:\GitHub\npy-matlab')

pathToYourConfigFile = 'D:\GitHub\KiloSort2\configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFileBench384.m'))

% common options for every probe
ops.chanMap     = 'D:\GitHub\KiloSort2\configFiles\neuropixPhase3A_kilosortChanMap.mat';
% ops.trange      = [3300 Inf]; % TIME RANGE IN SECONDS TO PROCESS
% ops.trange      = [3400 Inf]; % TIME RANGE IN SECONDS TO PROCESS
ops.trange      = [0 Inf]; % TIME RANGE IN SECONDS TO PROCESS

ops.Th       = 10;     % threshold on projections (like in Kilosort1)
ops.lam      = 40^2;   % weighting on the amplitude penalty (like in Kilosort1, but it has to be much larger)

ops.ThS      = [8 8];  % lower bound on acceptable single spike quality
ops.momentum = [20 400]; % number of samples to average over
ops.minFR = 1/50;

ops.sigmaMask  = 30;

ops.Nfilt       = 768; % max number of clusters
ops.nfullpasses = 1; % how many forward backward passes to do
ops.nPCs        = 3; % how many PCs to project the spikes into

ops.useRAM = 0;

dset = 'c46';

root = fullfile('H:\DATA\Spikes\', dset);
fname = sprintf('%s_aligned_npx_raw.bin', dset);
ops.fbinary     = fullfile(root, fname);
ops.fproc       = 'H:\DATA\Spikes\temp_wh.dat'; % residual from RAM of preprocessed data
ops.dir_rez     = 'H:\DATA\Spikes\';

% preprocess data
rez = preprocessDataSub(ops);

%% learnAndSolve7;
rez = learnAndSolve8(rez);

fGTname = sprintf('%s_spike_samples_npx.npy', dset);
sGT  = readNPY(fullfile('H:\DATA\Spikes\', fGTname));

testClu = rez.st3(:,2);
testRes = rez.st3(:,1);

gtClu = zeros(numel(sGT), 1, 'int32');
gtRes = sGT;


[allScores, allFPrates, allMissRates, allMerges] = ...
    compareClustering2(gtClu, gtRes, testClu, testRes, []);

savePath = fullfile('H:\DATA\Spikes\', dset);
mkdir(savePath)
rezToPhy(rez, savePath);

%%