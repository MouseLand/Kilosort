useGPU = 1; % do you have a GPU? Kilosorting 1000sec of 32chan simulated data takes 55 seconds on gtx 1080 + M2 SSD.

fpath    = 'F:\DATA\Spikes\eMouse\'; % where on disk do you want the simulation? ideally and SSD...
if ~exist(fpath, 'dir'); mkdir(fpath); end

% This part adds paths
addpath(genpath('D:\CODE\GitHub\KiloSort')) % path to kilosort folder
addpath(genpath('D:\CODE\GitHub\npy-matlab')) % path to npy-matlab scripts
pathToYourConfigFile = 'D:\CODE\GitHub\KiloSort\eMouse'; % for this example it's ok to leave this path inside the repo, but for your own config file you *must* put it somewhere else!  

% Run the configuration file, it builds the structure of options (ops)
run(fullfile(pathToYourConfigFile, 'config_eMouse.m'))

% This part makes the channel map for this simulation
make_eMouseChannelMap(fpath); 

% This part simulates and saves data. There are many options you can change inside this 
% function, if you want to vary the SNR or firing rates, or number of cells etc. 
% You can vary these to make the simulated data look more like your data.
% Currently it is set to relatively low SNR for illustration purposes in Phy. 
make_eMouseData(fpath, useGPU); 
%
% This part runs the normal Kilosort processing on the simulated data
[rez, DATA, uproj] = preprocessData(ops); % preprocess data and extract spikes for initialization
rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)

% This runs the benchmark script. It will report both 1) results for the
% clusters as provided by Kilosort (pre-merge), and 2) results after doing the best
% possible merges (post-merge). This last step is supposed to
% mimic what a user would do in Phy, and is the best achievable score
% without doing splits. 
benchmark_simulation(rez, fullfile(fpath, 'eMouseGroundTruth.mat'));

% save python results file for Phy
rezToPhy(rez, fpath);

fprintf('Kilosort took %2.2f seconds vs 72.77 seconds on GTX 1080 + M2 SSD \n', toc)

% now fire up Phy and check these results. There should still be manual
% work to be done (mostly merges, some refinements of contaminated clusters). 
%% AUTO MERGES 
% after spending quite some time with Phy checking on the results and understanding the merge and split functions, 
% come back here and run Kilosort's automated merging strategy. This block
% will overwrite the previous results and python files. Load the results in
% Phy again: there should be no merges left to do (with the default simulation), but perhaps a few splits
% / cleanup. On realistic data (i.e. not this simulation) there will be drift also, which will usually
% mean there are merges left to do even after this step. 
% Kilosort's AUTO merges should not be confused with the "best" merges done inside the
% benchmark (those are using the real ground truth!!!)

rez = merge_posthoc2(rez);
benchmark_simulation(rez, fullfile(fpath, 'eMouseGroundTruth.mat'));

% save python results file for Phy
rezToPhy(rez, fpath);

%% save and clean up
% save matlab results file for future use (although you should really only be using the manually validated spike_clusters.npy file)
save(fullfile(fpath,  'rez.mat'), 'rez', '-v7.3');

% remove temporary file
delete(ops.fproc);
%%
