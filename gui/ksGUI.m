

classdef ksGUI < handle
    % GUI for kilosort
    %
    % Purpose is to display some basic data and diagnostics, to make it easy to
    % run kilosort
    %
    % Kilosort by M. Pachitariu
    % GUI by N. Steinmetz
    %
    % TODO: 
    % - test that path adding and compilation work on a fresh install
    % - add a welcome/new user/help button
    % - indicate selected chans on probe view
    % - implement clicking on probe view
    % - save selected settings for next time
    % - make function to preprocess channel maps
    % - allow better setting of probe site shape/size
    

    properties        
        H % struct of handles to useful parts of the gui
        
        P % struct of parameters of the gui to remember, like last working directory        
        
        ops % struct for kilosort to run
        
        rez % struct of results of running
        
    end
    
    methods
        function obj = ksGUI(parent)
            
            obj.init();
            
            obj.build(parent);
            
            obj.initPars(); % happens after build since graphical elements are needed
            
        end
        
        function init(obj)
            
            % check that required functions are present
            if ~exist('uiextras.HBox')
                error('ksGUI:init:uix', 'You must have the "uiextras" toolbox to use this GUI. Choose Environment->Get Add-ons and search for "GUI Layout Toolbox" by David Sampson.\n')
            end
            
            % add paths
            mfPath = mfilename('fullpath');            
            if ~exist('readNPY')
                githubDir = fileparts(fileparts(fileparts(mfPath))); % taking a guess that they have a directory with all github repos
                if exist(fullfile(githubDir, 'npy-matlab'))
                    addpath(genpath(fullfile(githubDir, 'npy-matlab')));
                end
            end
            if ~exist('readNPY')
                warning('ksGUI:init:npy', 'In order to save data for phy, you must have the npy-matlab repository from https://github.com/kwikteam/npy-matlab in your matlab path\n');
            end
            
            % compile if necessary
            if ~exist('mexWtW2')
                
                fprintf(1, 'Compiled Kilosort files not found. Attempting to compile now.\n');
                try
                    oldDir = pwd;
                    cd(fullfile(fileparts(fileparts(mfPath)), 'CUDA'));
                    mexGPUall;
                    fprintf(1, 'Success!\n');
                    cd(oldDir);
                catch ex
                    fprintf(1, 'Compilation failed. Check installation instructions at https://github.com/cortex-lab/Kilosort\n');
                    rethrow(ex);
                end
            end
            
                     
                        
        end
        
        
        function build(obj, f)
            % construct the GUI with appropriate panels
            obj.H.fig = f;
            
            obj.H.root = uiextras.VBox('Parent', f,...
                'DeleteFcn', @(~,~)obj.cleanup(), 'Visible', 'on', ...
                'Padding', 5);
            
            % - Root sections
            obj.H.titleBar = uicontrol(...
                'Parent', obj.H.root,...
                'Style', 'text', 'HorizontalAlignment', 'left', ...
                'String', 'Kilosort', 'FontSize', 36,...
                'FontName', 'Myriad Pro', 'FontWeight', 'bold');
            
            obj.H.mainSection = uiextras.HBox(...
                'Parent', obj.H.root);
            
            obj.H.logPanel = uiextras.Panel(...
                'Parent', obj.H.root, ...
                'Title', 'Message Log', 'FontSize', 18,...
                'FontName', 'Myriad Pro');
            
            obj.H.root.Sizes = [-1 -8 -2];
            
            % -- Main section
            obj.H.setRunVBox = uiextras.VBox(...
                'Parent', obj.H.mainSection);
            
            obj.H.settingsPanel = uiextras.Panel(...
                'Parent', obj.H.setRunVBox, ...
                'Title', 'Settings', 'FontSize', 18,...
                'FontName', 'Myriad Pro');
            obj.H.runPanel = uiextras.Panel(...
                'Parent', obj.H.setRunVBox, ...
                'Title', 'Run', 'FontSize', 18,...
                'FontName', 'Myriad Pro');
            obj.H.setRunVBox.Sizes = [-2 -1];
            
            obj.H.probePanel = uiextras.Panel(...
                'Parent', obj.H.mainSection, ...
                'Title', 'Probe view', 'FontSize', 18,...
                'FontName', 'Myriad Pro', 'Padding', 10);
            
            obj.H.dataPanel = uiextras.Panel(...
                'Parent', obj.H.mainSection, ...
                'Title', 'Data view', 'FontSize', 18,...
                'FontName', 'Myriad Pro', 'Padding', 10);
            
            obj.H.mainSection.Sizes = [-1 -1 -2];
            
            % --- Settings panel
            obj.H.settingsVBox = uiextras.VBox(...
                'Parent', obj.H.settingsPanel);
            
            obj.H.settingsGrid = uiextras.Grid(...
                'Parent', obj.H.settingsVBox, ...
                'Spacing', 10, 'Padding', 5);
            
            % choose file
            obj.H.settings.ChooseFileTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'pushbutton', ...
                'String', 'Select data file', ...
                'Callback', @(~,~)obj.selectFileDlg);
                        
            % choose temporary directory
            obj.H.settings.ChooseTempdirTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'pushbutton', ...
                'String', 'Select working directory');
                        
            % choose output path
            obj.H.settings.ChooseOutputTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'pushbutton', ...
                'String', 'Select results output directory');
                        
            % choose probe
            obj.H.settings.setProbeTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Select probe layout');
            
            % set nChannels
            obj.H.settings.setnChanTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Number of channels');
                        
            % set Fs
            obj.H.settings.setFsTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Sampling frequency (Hz)');
                        
            % choose max number of clusters
            obj.H.settings.setNfiltTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Number of templates');
            
            % advanced options
            obj.H.settings.setAdvancedTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'pushbutton', ...
                'String', 'Set advanced options', ...
                'Callback', @(~,~)obj.advancedPopup());
            
            obj.H.settings.ChooseFileEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '...', 'Callback', @(~,~)obj.updateFileSettings());
            obj.H.settings.ChooseTempdirEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '...', 'Callback', @(~,~)obj.updateFileSettings());
            obj.H.settings.ChooseOutputEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '...', 'Callback', @(~,~)obj.updateFileSettings());
            obj.H.settings.setProbeEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'popupmenu', 'HorizontalAlignment', 'left', ...
                'String', {'Neuropixels Phase3A', '[new]', 'other...'}, ...
                'Callback', @(~,~)obj.updateProbeView());
            obj.H.settings.setnChanEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '', 'Callback', @(~,~)obj.updateFileSettings());
            obj.H.settings.setFsEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '30000', 'Callback', @(~,~)obj.updateFileSettings());            
            obj.H.settings.setNfiltEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '');
            
            set( obj.H.settingsGrid, ...
                'ColumnSizes', [-1 -1], 'RowSizes', -1*ones(1,8) );%
            
            obj.H.runVBox = uiextras.VBox(...
                'Parent', obj.H.runPanel,...
                'Spacing', 10, 'Padding', 5);
            
            % button for run
            obj.H.settings.runBtn = uicontrol(...
                'Parent', obj.H.runVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Run Kilosort', 'enable', 'off', ...
                'Callback', @(~,~)obj.run());
            
            % button for write script
            obj.H.settings.writeBtn = uicontrol(...
                'Parent', obj.H.runVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Write this run to script', ...
                'Callback', @(~,~)obj.writeScript());
            
            % button for save defaults
            obj.H.settings.savedefBtn = uicontrol(...
                'Parent', obj.H.runVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Save these as defaults', ...
                'Callback', @(~,~)obj.saveDefaults());
            
            % -- Probe view
            obj.H.probeAx = axes(obj.H.probePanel);
            set(obj.H.probeAx, 'ButtonDownFcn', @(f,k)obj.probeClickCB(f, k));
            hold(obj.H.probeAx, 'on');            
            
            % -- Data view
            obj.H.dataVBox = uiextras.VBox('Parent', ...
                obj.H.dataPanel, 'Padding', 20);
            
            obj.H.dataAx = axes(obj.H.dataVBox);   
            box(obj.H.dataAx, 'off');
            
            set(obj.H.fig, 'WindowScrollWheelFcn', @(src,evt)obj.scrollCB(src,evt))
            set(obj.H.fig, 'WindowButtonMotionFcn', @(src, evt)any(1));
            
            obj.H.timeAx = axes(obj.H.dataVBox);
            obj.H.dataVBox.Sizes = [-6 -1];
            
            
            
            % -- Message log
            obj.H.logBox = uicontrol(...
                'Parent', obj.H.logPanel,...
                'Style', 'listbox', 'Enable', 'inactive', 'String', {}, ...
                'Tag', 'Logging Display', 'FontSize', 14);
            
            obj.log('Initialization success.');
            
        end
        
        function initPars(obj)
            
            % get ops
            obj.ops = ksGUI.defaultOps();  
            
            % get gui defaults/remembered settings
            
            obj.P.currT = 0.1;
            obj.P.tWin = [0 0.1];
            obj.P.currY = 0;
            obj.P.currX = 0;
            obj.P.nChanToPlot = 16;
            obj.P.selChans = 1:16;
            obj.P.vScale = 0.1;
            obj.P.dataGood = false;
            obj.P.probeGood = false;
            
            obj.updateProbeView();
            
        end
        
        function selectFileDlg(obj)
            [filename, pathname] = uigetfile('*.*', 'Pick a data file.');
            
            if filename~=0 % 0 when cancel
                obj.H.settings.ChooseFileEdt.String = ...
                    fullfile(pathname, filename);
                obj.log(sprintf('Selected file %s', obj.H.settings.ChooseFileEdt.String));
                
                obj.P.lastDir = pathname;                
                obj.updateGUIdefs();
                
                obj.updateFileSettings();
            end
            
        end
        
        function updateFileSettings(obj)
            fprintf(1, 'update file settings\n');                        
            
            % check whether there's a data file and exists
            if ~exist(obj.H.settings.ChooseFileEdt.String)
                obj.log('Data file does not exist.');
                return;
            end
            
            % **check file extension
            
            % if data file exists and output/temp are empty, pre-fill
            if strcmp(obj.H.settings.ChooseTempdirEdt.String, '...')||...
                isempty(obj.H.settings.ChooseTempdirEdt.String)
                pathname = fileparts(obj.H.settings.ChooseFileEdt.String);
                obj.H.settings.ChooseTempdirEdt.String = pathname;
            end
            if strcmp(obj.H.settings.ChooseOutputEdt.String, '...')||...
                isempty(obj.H.settings.ChooseOutputEdt.String)
                pathname = fileparts(obj.H.settings.ChooseFileEdt.String);
                obj.H.settings.ChooseOutputEdt.String = pathname;
            end
            
            % if nChan is set, see whether it makes any sense
            if ~isempty(obj.H.settings.setnChanEdt.String)
                nChan = str2num(obj.H.settings.setnChanEdt.String);
                
                d = dir(obj.H.settings.ChooseFileEdt.String);
                b = d.bytes;
                
                bytesPerSamp = 2; % hard-coded for now, int16
                
                if mod(b,bytesPerSamp)==0 && mod(b/bytesPerSamp,nChan)==0
                    % if all that looks good, make the plot
                    obj.P.nSamp = b/bytesPerSamp/nChan;
                    obj.P.dataGood = true;
                    obj.updateDataView()                    
                else
                    obj.log('Doesn''t look like the number of channels is correct.');
                    
                    % try figuring it out
                    testNC = ceil(nChan*0.9):floor(nChan*1.1);
                    possibleVals = testNC(mod(b/bytesPerSamp, testNC)==0);
                    obj.log(sprintf('Consider trying: %s', num2str(possibleVals)));
                end
            end
                    
        end
        
        function updateGUIdefs(obj)
            % update the gui defaults to remember things the user has
            % chosen
            
        end
        
        function advancedPopup(obj)
            
            % bring up popup window to set other ops
            obj.log('setting advanced options not yet implemented.');
            
        end
        
        function run(obj)
            
            % check that everything is set up correctly to run
            
            % do preprocessing
            obj.ops.gui = obj; % for kilosort to access, e.g. calling "log"
            try
                obj.rez = preprocessDataSub(obj.ops);
            catch ex
                log(sprintf('Error preprocessing! %s', ex.message));
            end
            
            % update gui with results of preprocessing
            obj.updateDataView();
            
            % fit templates
            try
                obj.rez = learnAndSolve8(obj.rez);
            catch ex
                log(sprintf('Error running kilosort! %s', ex.message));
            end
            
            % save results
            try
                rezToPhy(obj.rez, obj.ops.saveDir);
            catch ex
                log(sprintf('Error saving data for phy! %s', ex.message));
            end
            
        end
        
        function updateDataView(obj)
            
            if obj.P.dataGood && obj.P.probeGood
                % get currently selected time and channels

                t = obj.P.currT;                
                chList = obj.P.selChans;
                tWin = obj.P.tWin;

                % show raw data traces
                if ~isfield(obj.P, 'datMMfile') || isempty(obj.P.datMMfile)
                    filename = obj.H.settings.ChooseFileEdt.String;
                    datatype = 'int16';
                    chInFile = str2num(obj.H.settings.setnChanEdt.String);                
                    nSamp = obj.P.nSamp;
                    mmf = memmapfile(filename, 'Format', {datatype, [chInFile nSamp], 'x'});
                    obj.P.datMMfile = mmf;
                    obj.P.datSize = [chInFile nSamp];
                else
                    mmf = obj.P.datMMfile;
                end

                Fs = str2double(obj.H.settings.setFsEdt.String);
                samps = ceil(Fs*(t+tWin));
                if all(samps>0 & samps<obj.P.datSize(2))
                    dat = mmf.Data.x(obj.ops.chanMap.chanMap(chList),samps(1):samps(2));
                           
                    if ~isfield(obj.H, 'dataTr') || numel(obj.H.dataTr)~=numel(chList)
                        % initialize traces
                        hold(obj.H.dataAx, 'off');
                        for q = 1:numel(chList)
                            obj.H.dataTr(q) = plot(obj.H.dataAx, 0, NaN, 'k');
                            hold(obj.H.dataAx, 'on');
                        end
                    end
                    
                    for q = 1:size(dat,1)
                        set(obj.H.dataTr(q), 'XData', (samps(1):samps(2))/Fs, ...
                            'YData', q+double(dat(q,:)).*obj.P.vScale);
                    end
                    
                end

                % if the preprocessing is complete, add whitened data

                % if kilosort is finished running, add residuals
                
            end
        end
        
        function scrollCB(obj,~,evt)
            disp(get(gcf,'CurrentModifier')))
            if obj.isInLims(obj.H.dataAx)
                % in data axis                
                if evt.VerticalScrollCount>0
                    obj.P.vScale = obj.P.vScale/1.2;
                elseif evt.VerticalScrollCount<0
                    obj.P.vScale = obj.P.vScale*1.2;
                end
                obj.updateDataView();
                
            elseif obj.isInLims(obj.H.probeAx)
                % in probe axis
                if obj.P.probeGood
                    cpP = get(obj.H.probeAx, 'CurrentPoint');
                    yl = get(obj.H.probeAx, 'YLim');
                    dyl = diff(yl);
                    currY = cpP(1,2);
                    yc = obj.ops.chanMap.ycoords;
                    mx = max(yc)+obj.ops.chanMap.siteSize/2;
                    mn = min(yc)-obj.ops.chanMap.siteSize/2;
                    if currY>mx; currY = mx; end
                    if currY<mn; currY = mn; end
                    if evt.VerticalScrollCount<0
                        dyl = dyl/1.2;
                    elseif evt.VerticalScrollCount>0
                        dyl = dyl*1.2;
                    end
                    
                    if dyl>(mx-mn); dyl = mx-mn; end
                    newyl = currY+[-0.5 0.5]*dyl;
                    if newyl(1)<min(yc)
                        newyl = newyl+min(yc)-newyl(1);
                    elseif newyl(2)>max(yc)
                        newyl = newyl+max(yc)-newyl(2);
                    end                    
                    set(obj.H.probeAx, 'YLim', newyl);
                end
            end
        end
        
        function updateProbeView(obj)
            obj.P.probeGood = false; % might have just selected a new one
            
            % if probe file exists, load it
            selProbe = obj.H.settings.setProbeEdt.String{obj.H.settings.setProbeEdt.Value};
            
            cm = [];
            switch selProbe
                case 'Neuropixels Phase3A'
                    cm = load('neuropixPhase3A_kilosortChanMap.mat');
                case '[new]'
                    obj.log('New probe creator not yet implemented.');
                    return;
                case 'other...'
                    [filename, pathname] = uigetfile('*.mat', 'Pick a channel map file.');
            
                    if filename~=0 % 0 when cancel
                        obj.log('choosing a different channel map not yet implemented.');
                        %q = load(fullfile(pathname, filename)); 
                        % ** check for all the right data, get a name for
                        % it, add to defaults                        
                    end
                    return;
            end
            
            if ~isempty(cm)
                % if it is valid, plot it
%                 set(obj.H.probeScat, 'XData', cm.xcoords, 'YData', cm.ycoords,...
%                     'SizeData', 20*double(cm.connected)+1);

                nSites = numel(cm.chanMap);
                ux = unique(cm.xcoords); uy = unique(cm.ycoords);
                
                if isfield(cm, 'siteSize') && ~isempty(cm.siteSize)
                    ss = cm.siteSize(1);
                else
                    ss = min(diff(uy)); 
                end
                cm.siteSize = ss;
                    
                
                if ~isfield(obj.H, 'probeSites') || isempty(obj.H.probeSites) || ...
                        numel(obj.H.probeSites)~=nSites
                    obj.H.probeSites = [];
                    hold(obj.H.probeAx, 'on');                    
                    sq = ss*([0 0; 0 1; 1 1; 1 0]-[0.5 0.5]);
                    for q = 1:nSites
                        obj.H.probeSites(q) = fill(obj.H.probeAx, sq(:,1), sq(:,2), 'g');
                        set(obj.H.probeSites(q), 'HitTest', 'off');
                    end
                    axis(obj.H.probeAx, 'equal');
                    set(obj.H.probeAx, 'XTick', [], 'YTick', []);
                    %axis(obj.H.probeAx, 'off');
                end
                
                y = obj.P.currY;
                x = obj.P.currX;
                nCh = obj.P.nChanToPlot;
                
                dists = ((cm.xcoords-x).^2 + (cm.ycoords-y).^2).^(0.5);
                [~, ii] = sort(dists);
                obj.P.selChans = ii(1:nCh);
                
                sq = ss*([0 0; 0 1; 1 1; 1 0]-[0.5 0.5]);
                for q = 1:nSites
                    set(obj.H.probeSites(q), 'XData', sq(:,1)+cm.xcoords(q), ...
                        'YData', sq(:,2)+cm.ycoords(q));
                    
                    if ismember(q, obj.P.selChans)
                        set(obj.H.probeSites(q), 'FaceColor', [0 0 1]);
                    else
                        set(obj.H.probeSites(q), 'FaceColor', [0 1 0]);
                    end
                end
                
                obj.ops.chanMap = cm;
                obj.P.probeGood = true;
                % if data file is also valid, compute RMS and plot that here
            end
        end
        
        function probeClickCB(obj, ~, keydata)
            obj.P.currX = round(keydata.IntersectionPoint(1));
            obj.P.currY = round(keydata.IntersectionPoint(2));
            obj.updateProbeView;
            obj.updateDataView;
        end
        
        function writeScript(obj)
            % write a .m file script that the user can use later to run
            % directly, i.e. skipping the gui
            
        end
        
        function cleanup(obj)
            fclose('all');
        end
        
        function log(obj, message)
            % show a message to the user in the log box
            timestamp = datestr(now, 'dd-mm-yyyy HH:MM:SS');
            str = sprintf('[%s] %s', timestamp, message);
            current = get(obj.H.logBox, 'String');
            set(obj.H.logBox, 'String', [current; str], ...
                'Value', numel(current) + 1);
        end
    end
    
    methods(Static)
        function ops = defaultOps()
            % look for a default ops file and load it
            if exist('defaultOps.mat')
                load('defaultOps.mat', 'ops');
            else
                ops = [];
            end
        end
        
        function docString = opDoc(opName)
            switch opName
                case 'NchanTOT'; docString = 'Total number of rows in the data file';
                case 'Th'; docString = 'Threshold on projections (like in Kilosort1)';
            end
        end
        
        function isIn = isInLims(ax)
            cp = get(ax, 'CurrentPoint');
            xl = get(ax, 'XLim');
            yl = get(ax, 'YLim');
            
            isIn = cp(1)>xl(1) && cp(1)<xl(2) && cp(1,2)>yl(1) && cp(1,2)<yl(2);
        end
        
    end
    
end


