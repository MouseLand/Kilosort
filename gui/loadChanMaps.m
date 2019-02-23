

function chanMaps = loadChanMaps()


ksroot = fileparts(fileparts(mfilename('fullpath')));
chanMapFiles = dir(fullfile(ksroot, 'configFiles', '*.mat'));

idx = 1;
chanMaps = [];
orig_state = warning;
warning('off','all')

for c = 1:numel(chanMapFiles)
    
    q = load(fullfile(ksroot, 'configFiles', chanMapFiles(c).name));
    
    cm = createValidChanMap(q, chanMapFiles(c).name);
    if ~isempty(cm)
        if idx==1; chanMaps = cm; else; chanMaps(idx) = cm; end       
        idx = idx+1;
    end
    
end

warning(orig_state);
