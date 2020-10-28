% benchmark manipulator
addpath(genpath('D:\GitHub\KiloSort2')) % path to kilosort folder
addpath('D:\GitHub\npy-matlab')

pathToYourConfigFile = 'D:\GitHub\KiloSort2\configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFile384.m'))
rootH = 'H:\';
ops.fproc       = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD

root0 = 'F:\Spikes\manipulator';
chanmaps = {'neuropixPhase3B1_kilosortChanMap_all.mat', 'NP2_kilosortChanMap.mat'};
for j = 1:2
   chanmaps{j} = fullfile(root0, chanmaps{j}); 
end

addpath('D:\Drive\CODE\KS2')
%%
dt = 1;
root0 = 'F:\Spikes\manipulator';
dr = [];
cc = [];
cc2 = {};
%%
for iprobe = 2
    for isess = [3]
        for ishift = [1 2 3]
            rootZ = fullfile(root0, sprintf('drift%d', isess), sprintf('p%d', iprobe));
            if ishift<3
                fname = fullfile(rootZ, sprintf('rez2_%d.mat', ishift-1));
                load(fname)
                 dr{iprobe, isess} = rez.row_shifts;
            else
                 fname = fullfile(rootZ, 'rez_datashift.mat');
                 load(fname)
            end
            
            bin = ops.fs * dt;
            st  = rez.st3(:,1);
            clu = rez.st3(:,2);
            nspikes = length(st);
            S = sparse(clu, ceil(st/bin), ones(nspikes,1));
            igood = get_good_units(rez);
            S = S(igood, :);
            nbins = size(S,2);
            
            root_drift = fullfile(root0, sprintf('drift%d', isess));
            dy = readNPY(fullfile(root_drift, 'manip.positions.npy'));
            ts = readNPY(fullfile(root_drift, sprintf('manip.timestamps_p%d.npy', iprobe)));
            dy_samps = interp1(ts, dy, dt/2 + [0:dt:dt*(nbins-1)]);
            dy_samps(isnan(dy_samps)) = 0;
            
            ix = ceil(ts(2)/dt):ceil(ts(end)/dt);
            ix2 = [1:ceil(ts(2)/dt)  ceil(ts(end)/dt):nbins];
            
            ix2(length(ix)+1:end) = [];
            
            cc{iprobe, isess, ishift}  = corr(dy_samps(ix)', S(:, ix)');
            cc2{iprobe, isess, ishift} = corr(dy_samps(ix(1:length(ix2)))', S(:, ix2)');            
        end
    end
end
%%
iprobe = 2;
isess = 3;
ishift = 1;
icc1 = cc{iprobe, isess, ishift};
icc2 = cc2{iprobe, isess, ishift};

cq = quantile(icc2, [.025, .975]);
NN = length(icc1);
nbad = sum(icc1<cq(1)) + sum(icc1>cq(2));
disp([NN-nbad, nbad, nbad/NN])
%%
csd = cellfun(@(x) mean(abs(x)), cc);
c2sd = cellfun(@(x) mean(abs(x)), cc2);
%%
sq(mean(csd, 2))
sq(mean(c2sd, 2))

%%
iprobe = 2;
isess = 3;
ishift = 1;
rootZ = fullfile(root0, sprintf('drift%d', isess), sprintf('p%d', iprobe));
fname = fullfile(rootZ, sprintf('rez2_%d.mat', ishift-1));
load(fname)
%%

imagesc(rez.ccb, [-5, 5])
%%
cellfun(@(x) numel(x), cc)



%%


