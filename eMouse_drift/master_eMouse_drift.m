useGPU = 1; % eMouse can run w/out GPU, but KS2 cannot.
useParPool = 0; % with current version, always faster wihtout parPool
makeNewData = 1; % set this to 0 to just resort a previously created data set
sortData = 1;
runBenchmark = 1; %set to 1 to compare sorted data to ground truth for the simulation

fpath    = 'C:\Users\labadmin\Documents\jic\050320_sim_test\'; % where on disk do you want the simulation? ideally an SSD...
if ~exist(fpath, 'dir'); mkdir(fpath); end

%KS2 path -- also has the waveforms for the simulation
KS2path = 'C:\Users\labadmin\Documents\jic\KS2_040920\Kilosort2\';

% add paths to the matlab path
addpath(genpath('C:\Users\labadmin\Documents\jic\KS2_040920\Kilosort2\')); % path to kilosort2 folder
addpath(genpath('C:\Users\labadmin\Documents\jic\npy-matlab-master\'));

% path to whitened, filtered proc file (on a fast SSD)
rootH = 'D:\kilosort_datatemp\';

% path to config file; if running the default config, no need to change.
pathToYourConfigFile = [KS2path,'eMouse_drift\']; % path to config file

% Create the channel map for this simulation; default is a small 64 site
% probe with imec 3A geometry.
NchanTOT = 64;
chanMapName = make_eMouseChannelMap_3B_short(fpath, NchanTOT);
ops.chanMap             = fullfile(fpath, chanMapName);

% Run the configuration file, it builds the structure of options (ops)

run(fullfile(pathToYourConfigFile, 'config_eMouse_drift_KS2.m'))


% This part simulates and saves data. There are many options you can change inside this 
% function, if you want to vary the SNR or firing rates, # of cells, etc.
% There are also parameters to set the amplitude and character of the tissue drift. 
% You can vary these to make the simulated data look more like your data
% or test the limits of the sorting with different parameters.

if( makeNewData )
    make_eMouseData_drift(fpath, KS2path, chanMapName, useGPU, useParPool);
end
%
% Run kilosort2 on the simulated data

if( sortData ) 
   
        % common options for every probe
        gpuDevice(1);   %re-initialize GPU
        ops.sorting     = 1; % type of sorting, 2 is by rastermap, 1 is old
        ops.NchanTOT    = NchanTOT; % total number of channels in your recording
        ops.trange      = [0 Inf]; % TIME RANGE IN SECONDS TO PROCESS


        ops.fproc       = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD

        % find the binary file in this folder
        rootZ     = fpath;
        ops.rootZ = rootZ;

        ops.fbinary     = fullfile(rootZ,  'sim_binary.imec.ap.bin');


        % preprocess data to create temp_wh.dat
        rez = preprocessDataSub(ops);

        % pre-clustering to re-order batches by depth
        rez = clusterSingleBatches(rez);
        
        % main optimization
        % learnAndSolve8;
        rez = learnAndSolve8b(rez);
        

        % final splits
        rez = find_merges(rez, 1);
        

        % final splits by SVD
        rez    = splitAllClusters(rez, 1);
        
        % final splits by amplitudes
        rez = splitAllClusters(rez, 0);
    
        % decide on cutoff
        rez = set_cutoff(rez);
         
        % this saves to Phy
        rezToPhy(rez, rootZ);

        % discard features in final rez file (too slow to save)
        rez.cProj = [];
        rez.cProjPC = [];
        
        fname = fullfile(rootZ, 'rezFinal.mat');
        save(fname, 'rez', '-v7.3');
        
        sum(rez.good>0)
        fileID = fopen(fullfile(rootZ, 'cluster_group.tsv'),'w');
        fprintf(fileID, 'cluster_id%sgroup', char(9));
        fprintf(fileID, char([13 10]));
        for k = 1:length(rez.good)
            if rez.good(k)
                fprintf(fileID, '%d%sgood', k-1, char(9));
                fprintf(fileID, char([13 10]));
            end
        end
        fclose(fileID);
        
        % remove temporary file
        delete(ops.fproc);
end


if runBenchmark
 load(fullfile(fpath, 'rezFinal.mat'));
 benchmark_drift_simulation(rez, fullfile(fpath, 'eMouseGroundTruth.mat'),...
     fullfile(fpath,'eMouseSimRecord.mat'),2,0, fullfile(fpath, 'output_cluster_metrics.txt'));
end
