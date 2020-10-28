function make_eMouseData_drift(fpath, KS2path, chanMapName, useGPU, useParPool)
% this script makes a binary file of simulated eMouse recording
% written by Jennifer Colonell, based on Marius Pachitariu's original eMouse simulator for Kilosort 1
% Adds the ability to simulate simple drift of the tissue relative to the
% probe sites.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% you can play with the parameters just below here to achieve a signal more similar to your own data!!! 
norm_amp  = 16.7; % if 0, use amplitudes of input waveforms; if > 0, set all amplitudes to norm_amp*rms_noise
mu_mean   = 0.75; % mean of mean spike amplitudes. Incoming waveforms are in uV; make <1 to make sorting harder
noise_model = 'gauss'; %'gauss' or 'fromData'; 'fromData' requires a noiseModel.mat built by make_noise_model
rms_noise = 10; % rms noise in uV. Will be added to the spike signal. 15-20 uV an OK estimate from real data
t_record  = 1200; % duration in seconds of simulation. longer is better (and slower!) (1000)
fr_bounds = [1 10]; % min and max of firing rates ([1 10])
tsmooth   = 0.5; % gaussian smooth the noise with sig = this many samples (increase to make it harder) (0.5)
chsmooth  = 0.5; % smooth the noise across channels too, with this sig (increase to make it harder) (0.5)
amp_std   = .1; % standard deviation of single spike amplitude variability (increase to make it harder, technically std of gamma random variable of mean 1) (.25)
fs_rec    = 30000; % sample rate for the for the recording. Waveforms must be sampled at a rate w/in .01% of this rate
nt        = 81; % number of timepoints expected. All waveforms must have this time window
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%drift params. See the comments in calcYPos_v2 for details
drift.addDrift = 1;
drift.sType = 'rigid';  %'rigid' or 'point';
drift.tType = 'sine';   %'exp' or 'sine'
drift.y0 = 3800;        %in um, position along probe where motion is largest
                        %y = 0 is the tip of the probe                        
drift.halfDistance = 1000;   %in um, distance along probe over which the motion decays
drift.amplitude = 10;        %in um for a sine wave
%                             peak variation is 2Xdrift.amplitude
drift.halfLife = 2;     %in seconds
drift.period = 600;      %in seconds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%waveform source data
% waveform files to use for the simulation
% these come in two types: 
%     3A data, used for "large" units that cover > 100 um in Y
%     Data from Kampff ultradense survey, analyzed by Nick Steimetz and
%     Susu Chen, for "small" units that cover < 100 um in Y
% Interpolation of large units uses a linear "scattered interpolant"
% Interpolation of small units uses a grid interpolant.
% currently, the signal at the simulated sites is taken as the interpolated
% field value at the center of the site. This is likely appropriate for
% mapping 3A data onto simulated 3A data (or 3B data). For creating 3A data
% from the Kampff data, averaging over site area might be more realistic.

%fileCopies specifies how many repeats of each file (to make dense data
%sets). The units will be placed at random x and y on the probe, but
%using many copies can lead to too many very similar units.

useDefault = 1;     %use waveforms from eMouse folder in KS2

if useDefault
    %get waveforms from eMouse folder in KS2
     filePath{1} = fullfile(KS2path,'eMouse_drift','kampff_St_unit_waves_allNeg_2X.mat');
     fileCopies(1) = 2;
     filePath{2} = fullfile(KS2path,'eMouse_drift','121817_SU_waves_allNeg_gridEst.mat');
     fileCopies(2) = 2;
else
    %fill in paths to waveform files 
    filePath = {};
    filePath{1} = 'C:\Users\labadmin\Documents\emouse_drift\eMouse\121817_single_unit_waves_allNeg.mat';
    fileCopies(1) = 1;
    filePath{2} = 'C:\Users\labadmin\Documents\emouse_drift\UltradenseKS4Jennifer\SpikeMeanWaveforms\kampff_St_unit_waves_allNeg_2X.mat';
    fileCopies(2) = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%other parameters
bPair     = 0; % set to 0 for randomly distributed units, 1 for units in pairs
pairDist  = 50; % distance between paired units
bPlot     = 0; %make diagnostic plots of waveforms

% Noise can either be generated from a gaussian distribution, or modeled on
% noise from real data, matching the frequency spectrum and cross channel
% correlation. The sample noise data is taken from a 3B2 recording performed
% by Susu Chen. Note that the noise data should come from a probe with
% the same geometry as the model probe.
if ( strcmp(noise_model,'fromData') )
    if useDefault
        nmPath = [KS2path,'\eMouse_drift\','SC026_3Bstag_noiseModel.mat'];
    else
        %fill in path to desired noise model.mat file
    end
    noiseFromData = load(nmPath);
end

% Add a SYNC channel to the file
% 16 bit word with a 1 Hz square wave in 7th bit
addSYNC = false; 
syncOffset = 0.232; % must be between 0 and 0.5, offset to first on edge
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng('default');

% There are 3 seeds for the randome number generator, so some parts of the
% simulation can be fixed while others vary from run to run

% For unit placment, average amplitude, and spike times

unit_seed = 101;

% For individual spike amplitudes, but still based on the same average
% Meant to simulate the same spikes showing up in different streams

amp_seed = 101;

% For noise generation

noise_seed = 101;


% set the seed of the random number generator used for unit definition
rng(unit_seed);  

%bitPerUV = 0.42667; %imec 3A or 3B, gain = 500
bitPerUV = 1.3107; %for NP 2.0, fixed gain of 80

%  load channel map file built with make_eMouseChannelMap_3A_short.m

chanMapFile = fullfile(fpath, chanMapName);
load(chanMapFile);

zeroSites = find(connected == 0);

Nchan = numel(chanMap); %physical sites on the probe
%invChanMap(chanMap) = [1:Nchan]; % invert the  channel map here--create the order in which to write output

% range of y positions to place units
minY = min(ycoords(find(connected==1)));
maxY = max(ycoords(find(connected==1)));
minX = min(xcoords(find(connected==1)));
maxX = max(xcoords(find(connected==1)));


%build the complete filePath list for waveforms
nUniqueFiles = numel(filePath);
currCount = nUniqueFiles;
for i = 1:nUniqueFiles
    for j = 1:fileCopies(i)-1
        currCount = currCount + 1;
        filePath{currCount} = filePath{i};
    end
end


nFile = length(filePath);

% for i = 1: nFile
%     fprintf('%s\n',filePath{i});
% end


NN = 0; %number of units included
uF = {}; %cell array for interpolants
uR = []; %array of max/min X and Y for points that can be interpolateed
         %structure with uR.maxX, uR.maxY, uR.minX, uR.minY
origLabel = []; %array of original labels


for fileIndex = 1:nFile
    % generate waveforms for the simulation
    uData = load(filePath{fileIndex});
    fs_diff = 100*(abs(uData.fs - fs_rec)/fs_rec);
    if (fs_diff > 0.01)
        fprintf( 'Waveform file %d has wrong sample rate.\n', fileIndex );
        fprintf( 'Skipping to next file.');
        continue
    end
    if (uData.nt ~= nt)
        fprintf( 'Waveform file %d has wrong time window.\n', fileIndex );
        fprintf( 'Skipping to next file.');
        continue
    end
    if ~( strcmp(uData.dataType, '3A') || strcmp(uData.dataType, 'UD') )
        fprintf( 'Waveform file %d has unknown sample type.\n', fileIndex );
        fprintf( 'Skipping to next file.');
        continue
    end
    
    
    nUnit = length(uData.uColl);
    unitRange = NN+1:NN+nUnit;  %for now, using all the units we have
    
    if strcmp(uData.dataType, '3A')
        uType(unitRange) = 0;
    end
    if strcmp(uData.dataType, 'UD')
        uType(unitRange) = 1;
    end
    
    %vary intensity for each unit
    if uType(NN+1) == 0
        scaleIntensity = mu_mean*(1 + (rand(nUnit,1) - 0.5));
    end
    if uType(NN+1) == 1     %these are already small, so don't make them smaller.
        scaleIntensity = (1 + (rand(nUnit,1)-0.5));
    end

   if (norm_amp > 0)
        %get min and max intensity, and normalize to norm_amp X the rms noise
        targAmp = rms_noise*norm_amp;
        for i = 1:nUnit          
            maxAmp = max(max(uData.uColl(i).waves)) - min(min(uData.uColl(i).waves));
            uData.uColl(i).waves = uData.uColl(i).waves*(targAmp/maxAmp);
        end
    else
        for i = 1:nUnit
            uData.uColl(i).waves = uData.uColl(i).waves*scaleIntensity(i);
        end
   end

    %deprecated
    %allowed use of scattered interpolants (non-grid sampling of the
    %waveforms -- but the calculations are 5-10X slower than gridded
    %interpolants. For non-square sampling of waveforms (e.g. Neuropixels
    %probes) -- make a scattered interpolant and then sample on a gridded
    %interpolant to feed to the simulator.
    %for these units, create an intepolant which will be used to
    %calculate the waveform at arbitrary sites
    
%     if (uType(unitRange(1)) == 1)
%         [uFcurr, uRcurr] = makeGridInt( uData.uColl, nt );
%     else
%         [uFcurr, uRcurr] = makeScatInt( uData.uColl, nt );
%     end
    
    % with both types gridded, always make a gridded interpolant
    [uFcurr, uRcurr] = makeGridInt( uData.uColl, nt );
    
    %append these to the array over all units
    uF = [uF, uFcurr];
    uR = [uR, uRcurr];
    
    for i = unitRange
        origLabel(i) = uData.uColl(i-NN).label;
    end
    
    NN = NN + nUnit;
end

% calculate a size for each unit
  uSize = zeros(1,NN);
  for i = 1:NN
      uSize(i) = (uR(i).maxX - uR(i).minX) * (uR(i).maxY - uR(i).minY);
  end
  
% distribute units along the length the probe, either in pairs separated
% by unitDist um, or fully randomly.
% for now, keep x = the original position (i.e. don't try to recenter)
uX = zeros(1,NN);
uY = zeros(1,NN);
uX = uX + ((maxX - minX)/2 + minX);

yRange = maxY - minY;

unitOrder = randperm(NN);   %really only important for the pairs

if ~bPair    
    uX(unitOrder) = (maxX - minX)/2 + minX;
    uY(unitOrder) = minY + yRange * rand(NN,1);
else
    %pairs will be placed at regular spacing (so the spacing is controlled)
    %want to avoid placing units at the ends of the probe
    endBuff = 50; %min distance in um from end of probe
    if (mod(NN,2))
        %odd number of units; pair up the first NN - 1
        %place the first units between (minY + pairDist) and maxY
        nPair = (NN-1)/2;
        % (0:nPair) = nPair + 1 positions; one extra for the loner
        pairPos = minY + endBuff + (pairDist/2) + ...
            (0:nPair)*(yRange - pairDist - 2*endBuff)/(nPair);
        uY(unitOrder(1:2:NN-2)) =  pairPos(1:nPair) + (pairDist/2);
        uY(unitOrder(2:2:NN-1)) = pairPos(1:nPair) - (pairDist/2);
        uY(unitOrder(NN)) = pairPos(nPair + 1);
    else
        %even number of units; pair them all
        nPair = NN/2;
        pairPos = minY + endBuff + (pairDist/2) + ...
            (0:nPair-1)*(yRange - pairDist - 2*endBuff)/(nPair-1);
        uY(unitOrder(1:2:NN-2)) = pairPos(1:nPair) + (pairDist/2);
        uY(unitOrder(2:2:NN-1)) = pairPos(1:nPair) - (pairDist/2); 
    end
end
    
% calculate monitor sites for these units
monSite = zeros(1,NN);
for i = 1:NN
    [currWav, uSites] = intWav( uF{i}, uX(i), uY(i), uR(i), xcoords, ycoords, connected, nt );
%         fprintf( 'site %d: ',i);
%         for uCount = 1:length(uSites)
%             fprintf( '%d,', uSites(uCount));         
%         end
%         fprintf( '\n' );
    monSite(i) = pickMonSite( currWav, uSites, connected );
    if( monSite(i) < 0 )
        fprintf( 'All sites nearby unit %d are not connected. Probably an error.\n', i);
        return;
    end
end


if (bPlot)
    
    %write out units and positions to console
    fprintf( 'label\torig label\ty position\tmonitor site\n' );
    for i = 1:NN
        fprintf( '%d\t%d\t%.1f\t%d\n', i, origLabel(i), uY(i), monSite(i) );
    end
    
    %for an overview of the units included, plot waves over the whole
    %probe, at the initial position, and shifted by 1*delta, and 2*delta
   
    deltaUM = 10;
    
    %set up waves for whole probe
    testwav = zeros(Nchan,nt);

    %for each unit, find the sites with signal, calculate the waveform based on
    %the current probe position (using the interpolant) and add to wav

    for i = 1:NN       
        [currWav, uSites] = intWav( uF{i}, uX(i), uY(i), uR(i), xcoords, ycoords, connected, nt );
        testwav(uSites,:) = testwav(uSites,:) + currWav;
    end

    
    %calculate waveforms with units shifted by deltaUM
    testwav2 = zeros(Nchan,nt);
    for i = 1:NN
        currYPos = uY(i) + deltaUM;
        [currWav, uSites] = intWav( uF{i}, uX(i), currYPos, uR(i), xcoords, ycoords, connected, nt );
        testwav2(uSites,:) = testwav2(uSites,:) + currWav;
    end
    
    %calculate waveforms with units shifted by 2*deltaUM
    testwav3 = zeros(Nchan,nt);
    
    for i = 1:NN
        currYPos = uY(i) + 2*deltaUM;
        [currWav, uSites] = intWav( uF{i}, uX(i), currYPos, uR(i), xcoords, ycoords, connected, nt );
        testwav3(uSites,:) = testwav3(uSites,:) + currWav;
    end
    figure(1);
    tRange = 1.1*nt;
    yRange = 1.1*( max(max(testwav)) - min(min(testwav)) );
    
    %properties of the 3A probe. 
    %TODO: add derivation of these from xcoords, ycoords
    xMin = 11;
    yMin = 20;
    xStep = 16;
    yStep = 20;
    
    tDat = 1:nt;
    %only plotting 1-384 (rather than 1-385) because 385 is unconnected 
    %digital channel
    
    for i = 1:Nchan-1
        currDat = testwav( i, : ); 
        currDat2 = testwav2(i,:);
        currDat3 = testwav3(i,:);
        xPlotPos = (xcoords(i) - xMin)/xStep;
        xOff = tRange * xPlotPos + nt/3;
        yOff = yRange * (ycoords(i) - yMin)/yStep;
        figure(1);
        %plot(tDat + xOff, currDat + yOff,'b-');
        %plot(tDat + xOff, scatDat + yOff,'b-');
        plot(tDat + xOff, currDat + yOff,'b-', tDat + xOff, currDat3 + yOff, 'r-' );
        if(xPlotPos == 0)
          msgstr=sprintf('ch%d',i);
          text(xOff,yOff+50,msgstr);  
        end
        hold on
    end
    
    %plot signals at monitor sites at initial position.
    figure(2);
    
    %some plotting params
    colorStr ={};
    colorStr{1} = '-b';
    colorStr{2} = '-r';
    %fprintf('unit\tmaxAmp\n');
    for i = 1:NN
        [currWav, uSites] = intWav( uF{i}, uX(i), uY(i), uR(i), xcoords, ycoords, connected, nt );
        cM = find(uSites == monSite(i));
        currDat = currWav(cM,:);
        %fprintf( '%d\t%.2f\n', i, (max(currDat)-min(currDat)));
        currColor = colorStr{uType(i) + 1};
        plot(tDat,currDat,currColor);
        hold on;
    end
    
    
end


bContinue = 1;

% set the sample rate to that specified in the hard coded params in this
% file (independent of fs read in through channel map)
% allows simulation of multiple streams with slightly different clock rates

fs = fs_rec;
fs_std = 30000; %used to generate spike times

%same for range of firing rates (note that we haven't included any info
%about the original firign rates of the units
fr = fr_bounds(1) + (fr_bounds(2)-fr_bounds(1)) * rand(NN,1); % create variability in firing rates

% totfr = sum(fr); % total firing rate

spk_times = [];
clu = [];

if bContinue        %done with setup, now starting the time consuming stuff
    
for j = 1:length(fr)      %loop over neurons
    %generate a set of time differences bewteen spikes
    %random numbers from a geometric distribution with with probability
    %(sample time)*firing rate = (1/sample rate)*firing rate =
    %(1/(samplerate*firingrate))
    %geometric distribution an appropriate model for the number of trials
    %before a success (neuron firing)
    %second two params for geornd are size of the array, here 2*firing
    %rate*total time of the simulation. 
    
    dspks = int64(geornd(1/(fs_std/fr(j)), ceil(2*fr(j)*t_record),1));
    dspks(dspks<ceil(fs_std * 2/1000)) = [];  % remove ISIs below the refractory period
    res = cumsum(dspks);
    spk_times = cat(1, spk_times, res);
    clu = cat(1, clu, j*ones(numel(res), 1));
end
% convert spike times to the requested rate
spk_times = int64( double(spk_times)*(fs/fs_std));

[spk_times, isort] = sort(spk_times);
clu = clu(isort);
clu       = clu(spk_times<t_record*fs);
spk_times = spk_times(spk_times<t_record*fs);
nspikes = numel(spk_times);

% this generates single spike amplitude with mean = 1 and std deviation
% ~amp_std, while ensuring all values are positive
% re-seed the random number generator to allow variable amplitudes
% while holding positions of units constant
rng(amp_seed);
amps = gamrnd(1/amp_std^2,amp_std^2, nspikes,1); 

%
buff = 128;
NT   = 4 * fs + buff; % batch size + buffer

fidW     = fopen(fullfile(fpath, 'sim_binary.imec.ap.bin'), 'w');

t_all    = 0;

if useGPU
    %set up the random number generators for run to run reproducibility 
    gpurng('default'); % start with a default set
    gpurng(noise_seed); % set the seed
    %gpurng('shuffle'); uncomment to have the seed set using the clock
end


% reset random number generator for noise
rng(noise_seed);  

%record a y position for each spike time
yDriftRec = zeros( length(spk_times), 5, 'double' );
allspks = 0;

% The parpool option can speed up the calculation when using scattered
% interpolants AND running with a large number of workers (>8). With all
% gridded interpolants, the overhead is too large and 
if (useParPool)
    %delete any currently running pool
    delete(gcp('nocreate'))
    %create parallel pool
    locCluster = parcluster('local');
    parpool('local',locCluster.NumWorkers);
    %get a handle to it
    p = gcp(); 
    %add variables to the workers
    p_xcoords = parallel.pool.Constant(xcoords);
    p_ycoords = parallel.pool.Constant(ycoords);
    p_connected = parallel.pool.Constant(connected);
    p_uF = parallel.pool.Constant(uF);
    p_uX = parallel.pool.Constant(uX);
    p_uY = parallel.pool.Constant(uY);
    p_uR = parallel.pool.Constant(uR);
end

while t_all<t_record
    
    if ( strcmp(noise_model,'gauss') )
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
        %rescale the smoothed data to make the std = 1;
        dat = zscore(dat, 1, 1);
        dat = gather_try(dat);
        %multiply the final noise calculation by the expected rms in uV
        dat = dat*rms_noise;
        %fprintf( 'Noise mean = %.3f; std = %.3f\n', mean(dat(:,1)), std(dat(:,1)));
    elseif ( strcmp(noise_model,'fromData') )        
        enoise = makeNoise( NT, noiseFromData, chanMap, connected, NchanTOT );
        if t_all>0
            enoise(1:buff, :) = enoise_old(NT-buff + [1:buff], :);
        end
        dat = enoise;
    end
    
    if addSYNC 
        sync = zeros(NT,1,'int16');
        hilo = zeros(NT,1,'logical');
        % for each time point, calculate where it is in the 1Hz cycle
        currTime = zeros(NT,1,'single');
        currTime(:,1) = (0:NT-1) + t_all*fs;
        currTime(:,1) = currTime(:,1)/fs - syncOffset;  %time in seconds, relative to offset
        hilo(:,1) = currTime(:,1) - floor(currTime(:,1)) < 0.5;
        sync(hilo,1) = 64;
    end
    
    if t_all>0
        dat(1:buff/2, :) = dat_old(NT-buff/2 + [1:buff/2], :);
    end
    
    
    % now we add spikes all channels; ref channels zeroed out after
    ibatch = (spk_times >= t_all*fs) & (spk_times < t_all*fs+NT-buff);
    ts = spk_times(ibatch) - t_all*fs;
    ids = clu(ibatch);
    am = amps(ibatch);
    tRange = int64(1:nt);
    tic
    
    if (useParPool)
        %     %first run a parfor loop for the time consuming calculation of the
        %     %interpolated waves
        
        currWavArray = zeros(length(ts),Nchan,nt,'double'); %actual array much shorter
        currSiteArray = zeros(length(ts),Nchan,'uint16'); %record indices of the sites in currWavArray;
        currNsiteArray = zeros(length(ts),'uint16'); %record number of sites
        
        parfor i = 1:length(ts)
            cc = ids(i); %current cluster index
            currT = t_all + double(ts(i))/fs;
            currYPos = calcYPos_v2( currT, p_uY.Value(cc), drift );
            [currWav, uSites] = ...
                intWav( p_uF.Value{cc}, p_uX.Value(cc), currYPos, p_uR.Value(cc), p_xcoords.Value, p_ycoords.Value, p_connected.Value, nt );
            nSite = length(uSites);
            currWavArray(i,:,:) = padarray(currWav * am(i),[(Nchan-nSite),0],0,'post');
            currSiteArray(i, :) = padarray(uSites,(Nchan-nSite),0,'post')';
            currNsiteArray(i) = nSite;
        end
        
    end
    
    for i = 1:length(ts)
        %for a given time, need to calcuate the wave that correspond to
        %the current position of the probe.
        allspks = allspks + 1;
        currT = t_all + double(ts(i))/fs;
        cc = ids(i); %current cluster index
        
        %get current position of this unit
        currYPos = calcYPos_v2( currT, uY(cc), drift );
        
        %currYdrift = calcYPos( currT, ycoords );
        yDriftRec(allspks,1) = currT;
        yDriftRec(allspks,2) = currYPos;
        yDriftRec(allspks,3) = cc; %record the unit label in the drift record
        
        
        if (useParPool)
            %get the waves for this unit from the big precalculated array
            uSites = squeeze(currSiteArray(i,1:currNsiteArray(i)));
            tempWav = squeeze(currWavArray(i,1:currNsiteArray(i),:));
            dat(ts(i) + tRange, uSites) = dat(ts(i) + tRange, uSites) + tempWav';
        else
            %calculate the interpolations now            
            [tempWav, uSites] = ...
                intWav( uF{cc}, uX(cc), currYPos, uR(cc), xcoords, ycoords, connected, nt );
            
            dat(ts(i) + tRange, uSites) = dat(ts(i) + tRange, uSites) +...
                am(i) * tempWav(:,:)';
        end
        
        cM = find(uSites == monSite(cc));
        %if drift has moved the monitor site outside the footprint of the
        %unit, the recorded amplitudes just stay = 0;
        
        if ( cM )
            yDriftRec(allspks,4) = max(tempWav(cM,:)) - min(tempWav(cM,:));
            yDriftRec(allspks,5) = max(dat(ts(i) + tRange,monSite(cc))) - min(dat(ts(i) + tRange,monSite(cc)));
        end
        
    end
    
    %zero out the unconnected channels

    dat(:, zeroSites') = 0; % these are the reference and dig channels
    
    dat_old    =  dat;
    %convert to 16 bit integers; waveforms are in uV
    dat = int16(bitPerUV * dat);
    if addSYNC
        %add the column of sync data
        dat = horzcat(dat, sync);
    end
    fwrite(fidW, dat(1:(NT-buff),:)', 'int16');
  
    t_all = t_all + (NT-buff)/fs;
    elapsedTime = toc;
    fprintf( 'created %.2f seconds of data; nSpikes %d; calcTime: %.3f\n ', t_all, length(ts), elapsedTime );
    enoise_old = enoise;
    clear currWavArray;
    clear currNSiteArray;
    clear currSiteArray;
end

fclose(fidW); % all done

gtRes = spk_times + nt/2; % add back the time of the peak for the templates (half the time span of the defined waveforms)
gtClu = clu;

save(fullfile(fpath, 'eMouseGroundTruth'), 'gtRes', 'gtClu')
save(fullfile(fpath, 'eMouseSimRecord'), 'yDriftRec' );

end

end

function [uF, uR] = makeScatInt( uColl, nt )
    %create an interpolating function for each unit in
    % the array uColl
    uF = {};
    % array of maximum radii for each unit. only sites within this radius
    % will get get contributions from this unit
    uR = [];
    for i = 1:length(uColl)
        %reshape waves to a 1D vector       
        wave_pts = reshape(uColl(i).waves',[uColl(i).nChan*nt,1]);      
        xpts = repelem(uColl(i).xC,nt)';
        ypts = repelem(uColl(i).yC,nt)';
        tpts = (repmat(1:nt,1,uColl(i).nChan))';
        %specify points as y, x to be congruent with row,column of grid
        %interpolant.
        uF{i} = scatteredInterpolant( ypts, xpts, tpts, wave_pts, 'linear', 'nearest' );
        uR(i).minX = min(uColl(i).xC);
        uR(i).maxX = max(uColl(i).xC);
        uR(i).minY = min(uColl(i).yC);
        uR(i).maxY = max(uColl(i).yC);
        %earlier version that used y extend
        %uR(i) = min( max(uColl(i).yC), abs(min(uColl(i).yC)) );  
    end
    
end

function [uF, uR] = makeGridInt( uColl, nt )
    %Reshape the array of [nsite, nt] into [Y, X, nt]
    %Note that this is dependent on how the data were stored:
    %here, assumes the sites are stored in row major order.
    %Could also derive this from the X and Y coordinates for generality
    
    %create an interpolating function for each unit in
    % the array uColl
    uF = {};
    % array of maximum radii for each unit. only sites within this radius
    % will get get contributions from this unit
    uR = [];

    for i = 1:length(uColl)         
        sCol = length(unique(uColl(i).xC));
        sRow = length(unique(uColl(i).yC));
        %reshape waveform array into [sRow x sCol x nt array]
        %the data are stored as (1,1),(1,2),(1,3)...(2,1),(2,3)
        %reshape transforms as "row fast", so need to reshape and permute
        v = permute(reshape(uColl(i).waves,[sCol,sRow,nt]),[2,1,3]);
        xVal = uColl(i).xC(1:sCol);
        %pick off first element of each column to get y values
        colOne = (0:sRow-1)*sCol + 1;
        yVal = uColl(i).yC(colOne);
        tVal = 1:nt;
        %remember: y = rows, x = columns!
        [Y,X,T] = ndgrid(yVal, xVal, tVal);
        uF{i} = griddedInterpolant(Y,X,T,v,'makima');
        uR(i).minX = min(uColl(i).xC);
        uR(i).maxX = max(uColl(i).xC);
        uR(i).minY = min(uColl(i).yC);
        uR(i).maxY = max(uColl(i).yC);
        %earlier version that just used y extent
        %uR(i) = min( max(uColl(i).yC), abs(min(uColl(i).yC)) );  
    end
    
end
 
function uSites = findSites( xPos, yPos, xcoords, ycoords, connected, uR )

    %find the sites currently in range for this unit at this point in time
    %(i.e. this value of yDrift)
    % xPos, yPos are the coordinates of the com of the unit signal in the 
    % coordinates of the probe at the current time
    % xcoords and ycoords are the positions of the sites, assumed constant
    % motion of the probe should be modeled as rigid motion of the units

    %calculate distance from xPos, yPos for each site
    xDist = xcoords - xPos;
    yDist = ycoords - yPos;
    
    uSites = find( (xDist >= uR.minX) & (xDist <= uR.maxX) & ...
                   (yDist >= uR.minY) & (yDist <= uR.maxY) & ...
           	       (connected==1) );
%     dist_sq = (xPos - xcoords).^2 + (yPos - (ycoords + yDrift)).^2;
%     % want to exclude sites that aren't connected; just add a constant so
%     % they won't pass the distance test
%     dist_sq(find(connected==0)) = dist_sq(find(connected==0)) + 10*maxRad^2;
%     uSites = find( dist_sq < maxRad^2 );
        
end

function [uWav, uSites] = intWav( currF, xPos, yPos, uR, xcoords, ycoords, connected, nt )


    % figure out for which sites we need to calculate the waveform
    uSites = findSites( xPos, yPos, xcoords, ycoords, connected, uR );

    % given an array of sites on the probe, calculate the waveform using
    % the interpolant determined for this unit  
    % xPos and yPos are the positions of the current unit
    % nt = number of timepoints
 
    currX = xcoords - xPos;
    currY = ycoords - yPos;
    
    nSites = length(uSites);
    xq = double(repelem(currX(uSites), nt));
    yq = double(repelem(currY(uSites), nt));
    tq = (double(repmat(1:nt, 1, nSites )))';
    %remember, y = rows in the grid, and x = columns in the grid
    %interpolation, and scattered interpolation set to match.
    nVal = numel(currF.Values);
    %tic
    uWav = currF( yq, xq, tq );
    %fprintf( '%d\t%d\t%.3f\n', numel(uSites), nVal, 1000*toc);
    uWav = (reshape(uWav', [nt,nSites]))';
    
end
function [uWav, uSites] = dumWav( currF, xPos, yPos, uR, xcoords, ycoords, connected, nt )

    % figure out for which sites we need to calculate the waveform
    uSites = findSites( xPos, yPos, xcoords, ycoords, connected, uR );
    uWav = zeros(length(uSites),nt);    
end

function monSite = pickMonSite( currWav, uSites, connected )

    % find the connected site with the largest expected signal with no
    % drift. If all uSites are unconnected, returns an error
    
    %calc amplitudes for each site
    currAmp = max(currWav,[],2) - min(currWav,[],2);
    
    %sort amplitudes
    [sortAmp, ind] = sort(currAmp,'descend');
    
    monSite = -1;
    i = 1;
    
    while ( i < length(uSites) && monSite < 0 )
        if (connected(uSites(ind(i))))
            monSite = uSites(ind(i));
        else
            i = i + 1;
        end
    end
        
end

function currYPos = calcYPos_v2( t, yPos0, drift  )
%   calculate current position of a unit given the current time and 
%   initial position (yPos0)

%   The pattern of tissue motion in space is set by drift.sType:
%       'rigid' -- all the units move together
%       'point' -- motion is largest at a point y0 (furthest from tip) and 
%         decreases exponentially for units far from y0. Need to
%         specify y0 and the halfDistance
%
%   The pattern of tissue motion in time is set by drift.tType:
%       'exp' -- initial fast transition followed by exponential
%            decay; specify stepsize (fast transition distance); halfLife (sec)
%            and period (in sec).
%       'sine' -- specify amplitude and period
%
%   Drift parameter values for 20 um p-p, uniform sine motion, with period = 300 s:    
%     drift.sType = 'rigid';        %'rigid' or 'point';
%     drift.tType = 'sine';          %'exp' or 'sine'
%     
%     drift.y0 = 3800;        %in um, position along probe where motion is largest
%                             %conventially, y = 0 is the bottom of the probe
%     drift.halfDistance = 1000;     %in um
%     
%     drift.amplitude = 10;         %in um. For a sine wave, the peak to
%                                    peak variation is 2Xdrift.amplitude
%     drift.halfLife = 10;          %in seconds
%     drift.period = 300;           %in seconds
%     
if (drift.addDrift)
    switch drift.tType
        case 'exp'
            timeIntoCycle = t - (floor(t/drift.period))*drift.period;
            delta = drift.amplitude*exp(-timeIntoCycle/drift.halfLife);
        case 'sine'          
            delta = drift.amplitude*sin(t*2*pi()/drift.period);
        otherwise
            fprintf( 'unknown parameter in drift calculation \n')
            return;
    end       
    
    switch drift.sType
        case 'rigid'
            %delta is equal for any position
        case 'point'
            %delta falls off exponentially from y0
            delta = delta * exp( -abs(drift.y0 - yPos0)/drift.halfDistance ); 
        otherwise
            fprintf( 'unknown parameter in drift calculation \n')
            return;
    end
    currYPos = yPos0 + delta;
else
    currYPos = yPos0;
end

end


function eNoise = makeNoise( noiseSamp,noiseModel,chanMap,connected,NchanTOT )

    %if chanMap is a short version of a 3A probe, use the first
    %nChan good channels to generate noise, then copy that array 
    %into an NT X NChanTot array
    
    nChan = numel(chanMap);        %number of noise channels to generate
    goodChan = sum(connected);
    tempNoise = zeros( noiseSamp, goodChan, 'single' );
    nT_fft = noiseModel.nm.nt;         %number of time points in the original time series
    fftSamp = noiseModel.nm.fft;
    
    noiseBatch = ceil(noiseSamp/nT_fft);    
    lastWind = noiseSamp - (noiseBatch-1)*nT_fft; %in samples
    
    for j = 1:goodChan     
            for i = 1:noiseBatch-1
                tStart = (i-1)*nT_fft+1;
                tEnd = i * nT_fft;            
                tempNoise(tStart:tEnd,j) = fftnoise(fftSamp(:,j),1);
            end
            %for last batch, call one more time and truncate
            lastBatch = fftnoise(fftSamp(:,j),1);
            tStart = (noiseBatch-1)*nT_fft+1;
            tEnd = noiseSamp;
            tempNoise(tStart:tEnd,j) = lastBatch(1:lastWind);             
    end
    
    %unwhiten this array
    Wrot = noiseModel.nm.Wrot(1:goodChan,1:goodChan);
    tempNoise_unwh = tempNoise/Wrot;
    
    %scale to uV; will get scaled back to bits at the end
    tempNoise_unwh = tempNoise_unwh/noiseModel.nm.bitPerUV;
    
    %to get the final noise array, map to an array including all channels
    eNoise = zeros(noiseSamp, NchanTOT, 'single');
    %indicies of the good channels
    goodChanIndex = find(connected);
    eNoise(:,chanMap(goodChanIndex)) = tempNoise_unwh;
    
end

function noise=fftnoise(f,Nseries)
% Generate noise with a given power spectrum.
% Useful helper function for Monte Carlo null-hypothesis tests and confidence interval estimation.
%  
% noise=fftnoise(f[,Nseries])
%
% INPUTS:
% f: the fft of a time series (must be a column vector)
% Nseries: number of noise series to generate. (default=1)
% 
% OUTPUT:
% noise: surrogate series with same power spectrum as f. (each column is a surrogate).
%
%   --- Aslak Grinsted (2009)
%  
if nargin<2
    Nseries=1;
end
f=f(:);     %ensures f is a column vector
N=length(f); 
Np=floor((N-1)/2);
phases=rand(Np,Nseries)*2*pi;
phases=complex(cos(phases),sin(phases)); % this was the fastest alternative in my tests. 
f=repmat(f,1,Nseries);
f(2:Np+1,:)=f(2:Np+1,:).*phases;
f(end:-1:end-Np+1,:)=conj(f(2:Np+1,:));
noise=real(ifft(f,[],1)); 

end



