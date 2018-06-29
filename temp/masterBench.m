addpath(genpath('D:\GitHub\KiloSort2')) % path to kilosort folder
addpath('D:\GitHub\npy-matlab')

rootZ = 'H:\DATA\Spikes\Benchmark2015';

pathToYourConfigFile = 'D:\GitHub\KiloSort2\configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFile384.m'))

ops.chanMap = 'H:\DATA\Spikes\Benchmark2015\forPRBimecToWhisper.mat';
rootH = 'H:\DATA\Spikes\temp\';

ops.NchanTOT = 129;
ops.trange      = [0 Inf]; % TIME RANGE IN SECONDS TO PROCESS

 % these settings overwrite any settings from config
ops.lam      = 10^2;   % weighting on the amplitude penalty (like in Kilosort1, but it has to be much larger)

ops.ThS      = [8 8];  % lower bound on acceptable single spike quality
ops.momentum = [20 400]; % number of samples to average over
ops.minFR    = 1/50; % minimum spike rate (Hz)

ops.sigmaMask  = 30;

ops.Nfilt       = 768; % max number of clusters
ops.nfullpasses = 1; % how many forward backward passes to do
ops.nPCs        = 3; % how many PCs to project the spikes into

ops.useRAM = 0;
ops.nNeighPC = 3;
ops.ccsplit = .975;

ops.NT                  = 64*1024+ ops.ntbuff;

dsets = {'20141202_all_es', '20150924_1_e', '20150601_all_s',...
    '20150924_1_GT', '20150601_all_GT', '20141202_all_GT'};

addpath(rootZ)

tic

for idk = [4 3 5] %:3 %1:length(dsets)
    fname = [dsets{idk} '.dat']; %'20141202_all_es.dat';
    
    savePath = fullfile(rootZ,  dsets{idk});
   
    ops.fbinary     = fullfile(savePath,  fname);
    ops.fproc       = fullfile(rootH, 'temp_wh2.dat'); % residual from RAM of preprocessed data
    
    % preprocess data
    rez = preprocessDataSub(ops);            
   
    fname = fullfile(savePath, 'rez.mat');
    if exist(fname, 'file')
        dr = load(fname);
        rez.iorig = dr.rez.iorig;
        rez.ccb = dr.rez.ccb;
        rez.ccbsort = dr.rez.ccbsort;
    else
        rez = clusterSingleBatches(rez);
        save(fname, 'rez', '-v7.3');
    end
    
    rez = learnAndSolve8b(rez);        
    
    rez = splitAllClusters(rez);
    
    fname = [dsets{idk} '.dat']; %'20141202_all_es.dat';
    [~, fn, ~] = fileparts(fname);
    
    isgood = rez.st3(:,4)<Inf;
    
    testClu = rez.st3(isgood,2);
    testRes = rez.st3(isgood,1);
     
    gtClu = LoadClu(fullfile(savePath, [fn '.clu.1']));
    fid   = fopen(fullfile(savePath, [fn '.res.1']), 'r');
    gtRes = int32(fscanf(fid, '%d'));
    fclose(fid);

    
    [allScores, allFPrates, allMissRates, allMerges] = ...
        compareClustering2(gtClu, gtRes, testClu, testRes, 1);
    
    rezToPhy(rez, savePath);

end

% save(fullfile(rootZ , 'allrez52718'), 'allrez', '-v7.3')

%%


%%