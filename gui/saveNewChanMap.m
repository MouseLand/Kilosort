

function saveNewChanMap(cm, obj)

newName = cm.name;
if strcmp(cm.name, 'unknown')
    answer = inputdlg('Name for this channel map:');
    if ~isempty(answer) && ~isempty(answer{1})
        newName = answer{1};
    else
        newName = '';
    end
end
if ~isempty(newName)
    ksRoot = fileparts(fileparts(mfilename('fullpath')));
    newFn = [answer{1} '_kilosortChanMap.mat'];
    save(fullfile(ksRoot, 'configFiles', newFn), '-struct', 'cm');
    obj.log(['Saved new channel map: ' fullfile(ksRoot, 'configFiles', newFn)]);
else
    obj.log('Could not save new channel map without a name');
end