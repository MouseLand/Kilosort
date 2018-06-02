

function [chanMap, xcoords, ycoords, kcoords, NchanTOT] = loadChanMap(cmIn)
% function [chanMap, xcoords, ycoords, kcoords] = loadChanMap(cmIn)
%
% Load and sanitize a channel map provided to Kilosort
%
% Inputs:
% - cmIn -
%       - is either a string, assumed to be a filepath to be loaded
%               containing the variables below, or
%       - a struct, containing fields:
%           - chanMap - the channel map, specifying which row of the data file
%                       contains each recorded and connected channel
%           - xcoords - a vector the same size as channel map, the physical
%                       position of each channel on the first dimension
%           - ycoords - a vector the same size as channel map, the physical
%                       position of each channel on the second dimension
%           - connected - optional - a logical vector the same size as channel map,
%                       indicating which channels are to be included.
%           - kcoords - optional - an integer vector the same size as
%                       chanMap, giving the group number for each channel
%
% Outputs:
%  - chanMap - the channel map, specifying which row of the data file
%              contains each recorded and connected channel
%  - xcoords - a vector of the position of each channel on dimension 1
%  - ycoords - a vector of the position of each channel on dimension 2
%  - kcoords - a vector of the group number of each channel
% 
% ops.Nchan should then be numel(chanMap)

if ischar(cmIn)    
    if exist(cmIn, 'file')        
        cmIn = load(cmIn);
    else
        error('ksLoadChanMap:FileNotFound', 'Could not find channel map file: %s', cmIn);
    end
end

if ~isfield(cmIn, 'chanMap') || isempty(cmIn.chanMap)
    error('ksLoadChanMap:NeedChanMap', 'The provided channel map must contain "chanMap"');
end
chanMap = cmIn.chanMap(:);

if ~isfield(cmIn, 'xcoords') || isempty(cmIn.xcoords) || numel(cmIn.xcoords)~=numel(chanMap)
    xcoords = zeros(size(chanMap)); % default to a vertical column
else
    xcoords = cmIn.xcoords(:);
end

if ~isfield(cmIn, 'ycoords') || isempty(cmIn.ycoords) || numel(cmIn.xcoords)~=numel(chanMap)
    ycoords = (1:numel(chanMap))'; % default to a vertical column
else
    ycoords = cmIn.ycoords(:);
end

if ~isfield(cmIn, 'kcoords') || isempty(cmIn.kcoords) || numel(cmIn.xcoords)~=numel(chanMap)
    kcoords = ones(size(chanMap)); % default to a single group
else
    kcoords = cmIn.kcoords(:);
end

if isfield(cmIn, 'connected') && ~isempty(cmIn.connected) 
    connected = logical(cmIn.connected(:));
    chanMap = chanMap(connected);
    xcoords = xcoords(connected);
    ycoords = ycoords(connected);
    
    kcoords = kcoords(connected);
    
    NchanTOT = sum(connected);
end

if ~exist('NchanTOT', 'var')
   NchanTOT = numel(chanMap); 
end

