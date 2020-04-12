function [chanMapName] = make_eMouseChannelMap_3B_short(fpath,NchanTOT)
% create a channel Map file for simulated data on a section of 
% an imec 3B probe to use with eMouse
% essentially identical to the 3A version

% total number of channels = 385 (in real 3A, 384 channels + digital) 
chanMap = (1:NchanTOT)';

% channels to ignore in analysis and when adding data
% in real 3B data, include the reference channels and the digital channel
% replicate the refChans that are within range of the short probe to
% preserve the geometry for channel to channel correlation in noise
% generation

allRef = [192];
refChan = allRef( find(allRef < max(NchanTOT)) );


connected = true(NchanTOT,1);
connected(refChan) = 0;

% copy the coordinates from MP chanmap for 3A. 
halfChan = floor(NchanTOT/2);

xcoords = zeros(NchanTOT,1,'double');
ycoords = zeros(NchanTOT,1,'double');

xcoords(1:4:NchanTOT) = 43;
xcoords(2:4:NchanTOT) = 11;
xcoords(3:4:NchanTOT) = 59;
xcoords(4:4:NchanTOT) = 27;

ycoords(1:2:NchanTOT) = 20*(1:halfChan);
ycoords(2:2:NchanTOT) = 20*(1:halfChan);


% Often, multi-shank probes or tetrodes will be organized into groups of
% channels that cannot possibly share spikes with the rest of the probe. This helps
% the algorithm discard noisy templates shared across groups. In
% this case, we set kcoords to indicate which group the channel belongs to.
% In our case all channels are on the same shank in a single group so we
% assign them all to group 1.
% Note that kcoords is not yet implemented in KS2 (08/15/2019)

kcoords = ones(NchanTOT,1);

% at this point in Kilosort we do data = data(connected, :), ycoords =
% ycoords(connected), xcoords = xcoords(connected) and kcoords =
% kcoords(connected) and no more channel map information is needed (in particular
% no "adjacency graphs" like in KlustaKwik). 
% Now we can save our channel map for the eMouse. 

% would be good to also save the sampling frequency here
fs = 30000; 

chanMapName = sprintf('chanMap_3B_%dsites.mat', NchanTOT);

save(fullfile(fpath, chanMapName), 'chanMap', 'connected', 'xcoords', 'ycoords', 'kcoords', 'fs' )