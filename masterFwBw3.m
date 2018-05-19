addpath(genpath('D:\GitHub\KiloSort2')) % path to kilosort folder
addpath('D:\GitHub\npy-matlab')

pathToYourConfigFile = 'D:\GitHub\KiloSort2\configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFileBench384.m'))

% common options for every probe
ops.chanMap     = 'D:\GitHub\KiloSort2\configFiles\neuropixPhase3A_kilosortChanMap_385.mat';
% ops.trange      = [3300 Inf]; % TIME RANGE IN SECONDS TO PROCESS
% ops.trange      = [3400 Inf]; % TIME RANGE IN SECONDS TO PROCESS
ops.trange      = [0 Inf]; % TIME RANGE IN SECONDS TO PROCESS

ops.Nfilt       = 512; % how many clusters to use
ops.nfullpasses = 6; % how many forward backward passes to do
ops.nPCs        = 7; % how many PCs to project the spikes into

ops.useRAM      = 0; % whether to use RAM for all data, or no data
ops.spkTh       = -4; % spike threshold
ops.nSkipCov    = 5; % how many batches to skip when computing whitening matrix

ops.lam = [5 40 40];
ops.NT          = 64*1024+ ops.ntbuff;% this is the batch size (try decreasing if out of memory) 

dset = 'c46';

root = fullfile('H:\DATA\Spikes\', dset);
fname = sprintf('%s_aligned_npx_raw.bin', dset);
ops.fbinary     = fullfile(root, fname);
ops.fproc       = 'H:\DATA\Spikes\temp_wh.dat'; % residual from RAM of preprocessed data
ops.dir_rez     = 'H:\DATA\Spikes\';

% preprocess data
rez = preprocessDataSub(ops);
%%
learnAndSolve5;
%
fGTname = sprintf('%s_spike_samples_npx.npy', dset);
sGT  = readNPY(fullfile('H:\DATA\Spikes\', fGTname));

% save the result file
%     fname = fullfile(ops.dir_rez,  sprintf('rez_%s_%s_%s.mat', mname, datexp, probeName{j}));
%     save(fname, 'rez', '-v7.3');

testClu = rez.st3(:,2);
testRes = rez.st3(:,1);

% ik = 371;
% testClu = rez.st3(rez.st3(:,2)==ik,2);
% testRes = rez.st3(rez.st3(:,2)==ik,1);
% testMu  = rez.st3(rez.st3(:,2)==ik,3);
% testClu = testClu(testMu>0);
% testRes = testRes(testMu>0);

% testClu = rez.clu(rez.clu>0);
% testRes = rez.st(rez.clu>0);

% gtClu = rez.clu(rez.clu==960);
% gtRes = rez.st(rez.clu==960);

gtClu = zeros(numel(sGT), 1, 'int32');
gtRes = sGT;

% clc
[allScores, allFPrates, allMissRates, allMerges] = ...
    compareClustering2(gtClu, gtRes, testClu, testRes, []);
%

[~, isort] = sort(rez.st3(:,1), 'ascend');
rez.st3 = rez.st3(isort, :);

rez.simScore = gather(WtW(:,:,ops.nt0));
rez.cProj    = fW(:, isort)';
rez.iNeigh   = gather(iList);

rez.ops = ops;

rez.W = Wall(:,:,:, round(nBatches/2));
rez.W = cat(1, zeros(nt0 - (ops.nt0-1-nt0min), Nfilt, Nrank), rez.W);

rez.U = Uall(:,:,:,round(nBatches/2));
rez.mu = muall(:,round(nBatches/2));

savePath = fullfile('H:\DATA\Spikes\', dset);
mkdir(savePath)
rezToPhy(rez, savePath);

%%