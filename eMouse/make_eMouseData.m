function make_eMouseData(fpath, useGPU)
% this script makes binary file of simulated eMouse recording

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% you can play with the parameters just below here to achieve a signal more similar to your own data!!! 
mu_mean   = 15; % mean of mean spike amplitudes. 15 should contain enough clustering errors to be instructive (in Phy). 20 is good quality data, 10 will miss half the neurons, 
nn        = 30; % number of simulated neurons (30)
t_record  = 1000; % duration in seconds of simulation. longer is better (and slower!) (1000)
fr_bounds = [1 10]; % min and max of firing rates ([1 10])
tsmooth   = 3; % gaussian smooth the noise with sig = this many samples (increase to make it harder) (3)
chsmooth  = 1; % smooth the noise across channels too, with this sig (increase to make it harder) (1)
amp_std   = .25; % standard deviation of single spike amplitude variability (increase to make it harder, technically std of gamma random variable of mean 1) (.25)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng('default');
rng(101);  % set the seed of the random number generator

dat = load('simulation_parameters'); % these are inside Kilosort repo
fs  = dat.fs; % sampling rate
wav = dat.waves; % mean waveforms for all neurons
wav = wav(:,:, randperm(size(wav,3), nn));

Nchan = numel(dat.xc) + 2; % we  add two fake dead channels
NN = size(wav,3); % number of neurons

chanMap = [33 34 8 10 12 14 16 18 20 22 24 26 28 30 32 ...
    7 9 11 13 15 17 19 21 23 25 27 29 31 1 2 3 4 5 6]; % this is the fake channel map I made

invChanMap(chanMap) = [1:34]; % invert the  channel map here

mu = mu_mean * (1 + (rand(NN,1) - 0.5)); % create variability in mean amplitude
fr = fr_bounds(1) + (fr_bounds(2)-fr_bounds(1)) * rand(NN,1); % create variability in firing rates

% totfr = sum(fr); % total firing rate

spk_times = [];
clu = [];
for j = 1:length(fr)
    dspks = int64(geornd(1/(fs/fr(j)), ceil(2*fr(j)*t_record),1));
    dspks(dspks<ceil(fs * 2/1000)) = [];  % remove ISIs below the refractory period
    res = cumsum(dspks);
    spk_times = cat(1, spk_times, res);
    clu = cat(1, clu, j*ones(numel(res), 1));
end
[spk_times, isort] = sort(spk_times);
clu = clu(isort);
clu       = clu(spk_times<t_record*fs);
spk_times = spk_times(spk_times<t_record*fs);
nspikes = numel(spk_times);

amps = gamrnd(1/amp_std^2,amp_std^2, nspikes,1); % this generates single spike amplitude variability of mean 1

%
buff = 128;
NT   = 4 * fs + buff; % batch size + buffer

fidW     = fopen(fullfile(fpath, 'sim_binary.dat'), 'w');

t_all    = 0;
while t_all<t_record
    if useGPU
        enoise = gpuArray.randn(NT, Nchan, 'single');
    else
        enoise = randn(NT, Nchan, 'single');
    end
    if t_all>0
        enoise(1:buff, :) = enoise_old(NT-buff + [1:buff], :);
    end
    
    dat = enoise;
    dat = my_conv2(dat, [tsmooth chsmooth], [1 2]);
    dat = zscore(dat, 1, 1);
    dat = gather_try(dat);
    
    if t_all>0
        dat(1:buff/2, :) = dat_old(NT-buff/2 + [1:buff/2], :);
    end
    
    dat(:, [1 2]) = 0; % these are the "dead" channels
    
    % now we add spikes on non-dead channels. 
    ibatch = (spk_times >= t_all*fs) & (spk_times < t_all*fs+NT-buff); 
    ts = spk_times(ibatch) - t_all*fs;
    ids = clu(ibatch);
    am = amps(ibatch);
    
    for i = 1:length(ts)
       dat(ts(i) + int64([1:82]), 2 + [1:32]) = dat(ts(i) + int64([1:82]), 2 + [1:32]) +...
           mu(ids(i)) * am(i) * wav(:,:,ids(i));
    end
    
    dat_old    =  dat;    
    dat = int16(200 * dat);
    fwrite(fidW, dat(1:(NT-buff),invChanMap)', 'int16');
    t_all = t_all + (NT-buff)/fs;
    
    enoise_old = enoise;
end

fclose(fidW); % all done

gtRes = spk_times + 42; % add back the time of the peak for the templates (answer to life and everything)
gtClu = clu;

save(fullfile(fpath, 'eMouseGroundTruth'), 'gtRes', 'gtClu')