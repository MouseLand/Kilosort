function k = kilosort
% Call this to open the kilosort gui. 

% add kilosort to path
mfPath = mfilename('fullpath');
if ~exist('preprocessDataSub')
    addpath(genpath(fileparts(mfPath)));
end

f = figure('Name', 'Kilosort',...
        'MenuBar', 'none',...
        'Toolbar', 'none',...
        'NumberTitle', 'off',...
        'Units', 'normalized',...
        'OuterPosition', [0.1 0.1 0.8 0.8]);
h = ksGUI(f);

if nargout > 0
  k = h;
end

end

