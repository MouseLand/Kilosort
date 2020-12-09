
addpath(genpath('D:\GitHub\KiloSort2')) % path to kilosort folder
addpath('D:\GitHub\npy-matlab')

pathToYourConfigFile = 'D:\GitHub\KiloSort2\configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFile384.m'))
rootH = 'H:\';
ops.fproc       = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD


ops.sig        = 20;  % spatial smoothness constant for registration
ops.fshigh     = 300; % high-pass more aggresively
ops.nblocks    = 5; % blocks for registration. 0 turns it off, 1 does rigid registration. Replaces "datashift" option.
% ops.AUCsplit   = .95;

datapath = {'G:\spikes\chronic\day2', 'G:\spikes\chronic\day1', ...
    'G:\spikes\chronic\day3', 'G:\spikes\chronic\day4'};

ops.chanMap = 'G:\Spikes\chronic\shank1_chanMap.mat';
        
ops.nblocks = 1;
ops.flag_warm_start = 0;
ops.trange = [0 Inf]; % time range to sort
ops.NchanTOT    = 96; % total number of channels in your recording

% ops.trange = [0 600]; % time range to sort

for j =  1:length(datapath)
    fprintf('Now running %s \n', datapath{j})
    if j==2
        rez_match = rez;
        ops.flag_warm_start = 1;   
    end
    
    % find the binary file in this folder
    rootZ = datapath{j};        
    
    % find the binary file
    fs          = [dir(fullfile(rootZ, '*.bin')) dir(fullfile(rootZ, '*.dat'))];
    ops.fbinary = fullfile(rootZ, fs(1).name);
    rootZ = fullfile(rootZ, 'kilosort');
    mkdir(rootZ);
    
    % preprocess data to create temp_wh.dat
    rez = preprocessDataSub(ops);
    
    if ops.flag_warm_start
        rez.F0 = rez_match.F0;
    end
    rez = datashaift2(rez, 1);
    
    if ops.flag_warm_start
        rez.W   = rez_match.W;
        rez.dWU = rez_match.dWU;
    end
    
    rez = learnAndSolve8b(rez, 1);    
   
    if ~getOr(ops, 'flag_warm_start', 0)
        rez = find_merges(rez, 1);
        rez = splitAllClusters(rez, 1);
    end
    
    % decide on cutoff
    rez = set_cutoff(rez);
    rez.good = get_good_units(rez);
    
    if ~getOr(ops, 'flag_warm_start', 0)
       rez = purge_rez(rez);
    end
    
    fprintf('found %d good units \n', sum(rez.good>0))
    
    % write to Phy
    fprintf('Saving results to Phy \n')
    rezToPhy(rez, rootZ, 1);
    
    % discard features in final rez file (too slow to save)
    rez.cProj = [];
    rez.cProjPC = [];
    
    
    % save final results as rez2
    fprintf('Saving final results in rez2\n')
    fname = fullfile(rootZ, 'rez_datashift.mat');
    save(fname, 'rez'); %, '-v7.3'
    
end
%% if you want to save the results to a Matlab file... 

