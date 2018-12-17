function k = kilosort
% Call this to open the kilosort gui. 

% add kilosort to path
if ~exist('ksGUI', 'file')
    mfPath = mfilename('fullpath');
    addpath(genpath(fileparts(mfPath)));
end

f = figure(1029321); % ks uses some defined figure numbers for plotting - with this random number we don't clash
set(f,'Name', 'Kilosort',...
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

