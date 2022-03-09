% main data
directory = 'C:/paht/to/Imro-files';
animal = 'James_Dean';

%%
% read imro file
imroData = readmatrix(fullfile(directory, ['LongCol_1shank_' animal '.imro']), 'FileType','text','Delimiter',')(','OutputType','string');
channel_list = zeros(length(imroData)-1,1);
bank = zeros(length(imroData)-1,1);
for i = 1:length(imroData)-1
    c = double(split(imroData(i+1),' '));
    channel_list(i) = c(1) + 1;
    bank(i) = c(2);
end

%%
% create a channel map
pos = [43, 27, 59, 11]; % possible x-positions on shank
Nchannels = 385; % if there is a sync channel, otherwise 384
connected = true(Nchannels, 1);
connected([192,385], 1) = false; % exclude reference and sync channel
chanMap = 1:Nchannels;
chanMap0ind = chanMap - 1;
xcoords = zeros(Nchannels,1);
ycoords = zeros(Nchannels,1);
for i = 1:length(channel_list)
    ycoords(channel_list(i),1) = round((channel_list(i)+bank(i)*(Nchannels-1))/2) * 20;
    xcoords(channel_list(i),1) = pos(mod(channel_list(i),4)+1);
end
shankInd   = ones(Nchannels,1);

fs = 30000; % sampling frequency

%%
% save data in a mat file
save(fullfile(directory, ['chanMap_' name '.mat']), 'chanMap','connected', 'xcoords', 'ycoords', 'shankInd', 'chanMap0ind', 'name', 'fs')
