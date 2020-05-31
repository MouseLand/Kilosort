

classdef ksGUI < handle
    % GUI for kilosort
    %
    % Purpose is to display some basic data and diagnostics, to make it easy to
    % run kilosort
    %
    % Kilosort by M. Pachitariu
    % GUI by N. Steinmetz
    %
    % TODO: (* = before release)
    % - allow better setting of probe site shape/size
    % - auto-load number of channels from meta file when possible
    % - update time plot when scrolling in dataview
    % - show RMS noise level of channels to help selecting ones to drop?
    % - implement builder for new probe channel maps (cm, xc, yc, name,
    % site size)
    % - saving of probe layouts
    % - plotting bug: zoom out on probe view should allow all the way out
    % in x
    % - some help/tools for working with other datafile types
    % - update data view needs refactoring... load a bigger-than-needed
    % segment of data, and just move around using this as possible
    % - when re-loading, check whether preprocessing can be skipped 
    % - find way to run ks in the background so gui is still usable(?)
    % - quick way to set working/output directory when selecting a new file
    % - when selecting a new file, reset everything
    % - why doesn't computeWhitening run on initial load?

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
                error('ksGUI:init:uix', 'You must have the "uiextras" toolbox to use this GUI. Choose Home->Add-Ons->Get Add-ons and search for "GUI Layout Toolbox" by David Sampson. You may have to search for the author''s name to find the right one for some reason. If you cannot find it, go here to download: https://www.mathworks.com/matlabcentral/fileexchange/47982-gui-layout-toolbox\n')
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
            if ~exist('mexSVDsmall2')
                
                fprintf(1, 'Compiled Kilosort files not found. Attempting to compile now.\n');
                try
                    oldDir = pwd;
                    cd(fullfile(fileparts(fileparts(mfPath)), 'CUDA'));
                    mexGPUall;
                    fprintf(1, 'Success!\n');
                    cd(oldDir);
                catch ex
                    fprintf(1, 'Compilation failed. Check installation instructions at https://github.com/MouseLand/Kilosort2\n');
                    rethrow(ex);
                end
            end
            
            obj.P.allChanMaps = loadChanMaps();         
                        
        end
        
        
        function build(obj, f)
            % construct the GUI with appropriate panels
            obj.H.fig = f;
            set(f, 'UserData', obj);
            
            set(f, 'KeyPressFcn', @(f,k)obj.keyboardFcn(f, k));
            
            obj.H.root = uiextras.VBox('Parent', f,...
                'DeleteFcn', @(~,~)obj.cleanup(), 'Visible', 'on', ...
                'Padding', 5);
            
            % - Root sections
            
            obj.H.titleHBox = uiextras.HBox('Parent', obj.H.root, 'Spacing', 50);                        
            
            obj.H.mainSection = uiextras.HBox(...
                'Parent', obj.H.root);
            
            obj.H.logPanel = uiextras.Panel(...
                'Parent', obj.H.root, ...
                'Title', 'Message Log', 'FontSize', 18,...
                'FontName', 'Myriad Pro');
            
            obj.H.root.Sizes = [-1 -12 -2];
            
            % -- Title bar
            
            obj.H.titleBar = uicontrol(...
                'Parent', obj.H.titleHBox,...
                'Style', 'text', 'HorizontalAlignment', 'left', ...
                'String', 'Kilosort', 'FontSize', 36,...
                'FontName', 'Myriad Pro', 'FontWeight', 'bold');
            
            obj.H.helpButton = uicontrol(...
                'Parent', obj.H.titleHBox,...
                'Style', 'pushbutton', ...
                'String', 'Help', 'FontSize', 24,...
                'Callback', @(~,~)obj.help);
            
            obj.H.resetButton = uicontrol(...
                'Parent', obj.H.titleHBox,...
                'Style', 'pushbutton', ...
                'String', 'Reset GUI', 'FontSize', 24,...
                'Callback', @(~,~)obj.reset);
            
            obj.H.titleHBox.Sizes = [-5 -1 -1];
            
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
            obj.H.setRunVBox.Sizes = [-4 -1];
            
            obj.H.probePanel = uiextras.Panel(...
                'Parent', obj.H.mainSection, ...
                'Title', 'Probe view', 'FontSize', 18,...
                'FontName', 'Myriad Pro', 'Padding', 5);
            
            obj.H.dataPanel = uiextras.Panel(...
                'Parent', obj.H.mainSection, ...
                'Title', 'Data view', 'FontSize', 18,...
                'FontName', 'Myriad Pro', 'Padding', 5);
            
            obj.H.mainSection.Sizes = [-1 -1 -3];
            
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
                'String', 'Select working directory', ...
                'Callback', @(s,~)obj.selectDirDlg(s),...
                'Tag', 'temp');
                        
            % choose output path
            obj.H.settings.ChooseOutputTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'pushbutton', ...
                'String', 'Select results output directory', ...
                'Callback', @(s,~)obj.selectDirDlg(s),...
                'Tag', 'output');
                        
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
                        
            obj.H.settings.setTrangeTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Time range (s)');
            
            % good channels
            obj.H.settings.setMinfrTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Min. firing rate per chan (0=include all chans)');
            
            % choose threshold
            obj.H.settings.setThTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Threshold');

            %                         lambda
            obj.H.settings.setLambdaTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Lambda');
            
            % ccsplit
            obj.H.settings.setCcsplitTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'AUC for splits');            
            
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
            
            probeNames = {obj.P.allChanMaps.name};
            probeNames{end+1} = '[new]'; 
            probeNames{end+1} = 'other...'; 
            obj.H.settings.setProbeEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'popupmenu', 'HorizontalAlignment', 'left', ...
                'String', probeNames, ...
                'Callback', @(~,~)obj.updateProbeView('reset'));
            obj.H.settings.setnChanEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '', 'Callback', @(~,~)obj.updateFileSettings());
            obj.H.settings.setFsEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '30000', 'Callback', @(~,~)obj.updateFileSettings());            
            obj.H.settings.setTrangeEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '');
            obj.H.settings.setMinfrEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '');
            obj.H.settings.setThEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '');
            obj.H.settings.setLambdaEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '');
            obj.H.settings.setCcsplitEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '');
            
            
            set( obj.H.settingsGrid, ...
                'ColumnSizes', [-1 -1], 'RowSizes', -1*ones(1,12) );% 12?
            
            obj.H.runVBox = uiextras.VBox(...
                'Parent', obj.H.runPanel,...
                'Spacing', 10, 'Padding', 5);
            
            % button for run
            obj.H.runHBox = uiextras.HBox(...
                'Parent', obj.H.runVBox,...
                'Spacing', 10, 'Padding', 5);
            
            obj.H.settings.runBtn = uicontrol(...
                'Parent', obj.H.runHBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Run All', 'enable', 'off', ...
                'FontSize', 20,...
                'Callback', @(~,~)obj.runAll());
            
            obj.H.settings.runEachVBox = uiextras.VBox(...
                'Parent', obj.H.runHBox,...
                'Spacing', 3, 'Padding', 3);
            
            obj.H.runHBox.Sizes = [-3 -1];
            
            obj.H.settings.runPreprocBtn = uicontrol(...
                'Parent', obj.H.settings.runEachVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Preprocess', 'enable', 'off', ...
                'Callback', @(~,~)obj.runPreproc());
            
            obj.H.settings.runSpikesortBtn = uicontrol(...
                'Parent', obj.H.settings.runEachVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Spikesort', 'enable', 'off', ...
                'Callback', @(~,~)obj.runSpikesort());
            
            obj.H.settings.runSaveBtn = uicontrol(...
                'Parent', obj.H.settings.runEachVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Save for Phy', 'enable', 'off', ...
                'Callback', @(~,~)obj.runSaveToPhy());
            
            % button for write script
%             obj.H.settings.writeBtn = uicontrol(...
%                 'Parent', obj.H.runVBox,...
%                 'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
%                 'String', 'Write this run to script', ...
%                 'Callback', @(~,~)obj.writeScript());
            
            % button for save defaults
%             obj.H.settings.saveBtn = uicontrol(...
%                 'Parent', obj.H.runVBox,...
%                 'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
%                 'String', 'Save state', ...
%                 'Callback', @(~,~)obj.saveGUIsettings());
            
            %obj.H.runVBox.Sizes = [-3 -1 -1];
            obj.H.runVBox.Sizes = [-1];
            
            % -- Probe view
            obj.H.probeAx = axes(obj.H.probePanel);
            set(obj.H.probeAx, 'ButtonDownFcn', @(f,k)obj.probeClickCB(f, k));
            hold(obj.H.probeAx, 'on');            
            
            % -- Data view
            obj.H.dataVBox = uiextras.VBox('Parent', ...
                obj.H.dataPanel, 'Padding', 20);
            
            obj.H.dataControlsTxt = uicontrol('Parent', obj.H.dataVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Controls',...
                'FontWeight', 'bold', ...
                'Callback', @(~,~)helpdlg({'Controls:','','--------','',...
                ' [1, 2, 3, 4] Enable different data views','',...
                ' [c] Toggle colormap vs traces mode', '',...
                ' [up, down] Add/remove channels to be displayed', '',...
                ' [scroll and alt/ctrl/shift+scroll] Zoom/scale/move ', '',...
                ' [click] Jump to position and time', '',...
                ' [right click] Disable nearest channel'}));
            
            obj.H.dataAx = axes(obj.H.dataVBox);   
            
            set(obj.H.dataAx, 'ButtonDownFcn', @(f,k)obj.dataClickCB(f, k));
            hold(obj.H.probeAx, 'on');
            
            set(obj.H.fig, 'WindowScrollWheelFcn', @(src,evt)obj.scrollCB(src,evt))
            set(obj.H.fig, 'WindowButtonMotionFcn', @(src, evt)any(1));
            
            obj.H.timeAx = axes(obj.H.dataVBox);
            
            sq = [0 0; 0 1; 1 1; 1 0];
            obj.H.timeBckg = fill(obj.H.timeAx, sq(:,1), sq(:,2), [0.3 0.3 0.3]);
            hold(obj.H.timeAx, 'on');
            obj.H.timeLine = plot(obj.H.timeAx, [0 0], [0 1], 'r', 'LineWidth', 2.0);
            title(obj.H.timeAx, 'time in recording - click to jump');
            axis(obj.H.timeAx, 'off');
            set(obj.H.timeBckg, 'ButtonDownFcn', @(f,k)obj.timeClickCB(f,k));
            
            obj.H.dataVBox.Sizes = [30 -6 -1];
            
            
            
            % -- Message log
            obj.H.logBox = uicontrol(...
                'Parent', obj.H.logPanel,...
                'Style', 'listbox', 'Enable', 'inactive', 'String', {}, ...
                'Tag', 'Logging Display', 'FontSize', 14);                        
        end
        
        function initPars(obj)
            
            % get ops
            obj.ops = ksGUI.defaultOps();  
            
            obj.P.currT = 0.1;
            obj.P.tWin = [0 0.1];
            obj.P.currY = 0;
            obj.P.currX = 0;
            obj.P.nChanToPlot = 16;
            obj.P.nChanToPlotCM = 16;
            obj.P.selChans = 1:16;            
            obj.P.vScale = 0.005;
            obj.P.dataGood = false;
            obj.P.probeGood = false;            
            obj.P.ksDone = false;
            obj.P.colormapMode = false; 
            obj.P.showRaw = true;
            obj.P.showWhitened = false;
            obj.P.showPrediction = false;
            obj.P.showResidual = false;
            
            mfPath = fileparts(mfilename('fullpath'));
            cm = load(fullfile(mfPath, 'cmap.mat')); %grey/red
            obj.P.colormap = cm.cm; 
            
            % get gui defaults/remembered settings
            obj.P.settingsPath = fullfile(mfPath, 'userSettings.mat');
            if exist(obj.P.settingsPath, 'file')
                savedSettings = load(obj.P.settingsPath);
                if isfield(savedSettings, 'lastFile')
                    obj.H.settings.ChooseFileEdt.String = savedSettings.lastFile;
                    obj.log('Initializing with last used file.');
                    try
                        obj.restoreGUIsettings();
                        obj.updateProbeView('new');
                        obj.updateFileSettings();
                    catch ex
                        obj.log('Failed to initialize last file.');
%                         keyboard
                    end
                end
            else
                obj.log('Select a data file (upper left) to begin.');
            end
            
%             obj.updateProbeView('new');
            
            
            
        end
        
        function selectFileDlg(obj)
            [filename, pathname] = uigetfile('*.*', 'Pick a data file.');
            
            if filename~=0 % 0 when cancel
                obj.H.settings.ChooseFileEdt.String = ...
                    fullfile(pathname, filename);
                obj.log(sprintf('Selected file %s', obj.H.settings.ChooseFileEdt.String));
                
                obj.P.ksDone = false;
                
                obj.updateFileSettings();
            end
            
        end
        
        function selectDirDlg(obj, src)
            switch src.Tag
                case 'output'
                    startDir = obj.H.settings.ChooseOutputEdt.String;
                case 'temp'
                    startDir = obj.H.settings.ChooseTempdirEdt.String;
            end
            if strcmp(startDir, '...'); startDir = ''; end
            pathname = uigetdir(startDir, 'Pick a directory.');
            
            if pathname~=0 % 0 when cancel
                switch src.Tag
                    case 'output'
                        obj.H.settings.ChooseOutputEdt.String = pathname;
                    case 'temp'
                        obj.H.settings.ChooseTempdirEdt.String = pathname;
                end
            end
            obj.updateFileSettings();
        end
        
        function updateFileSettings(obj)
            
            % check whether there's a data file and exists
            if strcmp(obj.H.settings.ChooseFileEdt.String, '...')
                return;
            end
            if ~exist(obj.H.settings.ChooseFileEdt.String, 'file')                     
                obj.log('Data file does not exist.');
                return;
            end
            
            
            % check file extension
            [~,~,ext] = fileparts(obj.H.settings.ChooseFileEdt.String);
            if ~strcmp(ext, '.bin') &&  ~strcmp(ext, '.dat')
                obj.log('Warning: Data file must be raw binary. Other formats not supported.');
            end
            
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
            
            nChan = obj.checkNChan();                    
                
            if ~isempty(nChan)
                % if all that looks good, make the plot
            
                obj.P.dataGood = true;
                obj.P.datMMfile = [];
                if nChan>=64
                    obj.P.colormapMode = true;
                    obj.P.nChanToPlotCM = nChan;
                end
                obj.updateDataView()

                lastFile = obj.H.settings.ChooseFileEdt.String;
                save(obj.P.settingsPath, 'lastFile');

                if obj.P.probeGood
                    set(obj.H.settings.runBtn, 'enable', 'on');
                    set(obj.H.settings.runPreprocBtn, 'enable', 'on');
                end      
            end
            obj.refocus(obj.H.settings.ChooseFileTxt);
            
        end
        
        function nChan = checkNChan(obj)
            origNChan = 0;
            % if nChan is set, see whether it makes any sense
            if ~isempty(obj.H.settings.setnChanEdt.String)
                nChan = str2num(obj.H.settings.setnChanEdt.String);
                origNChan = nChan;
                if isfield(obj.P, 'chanMap') && sum(obj.P.chanMap.connected)>nChan
                    nChan = numel(obj.P.chanMap.chanMap); % need more channels
                end
                    
            elseif isfield(obj.P, 'chanMap')  
                % initial guess that nChan is the number of channels in the channel
                % map
                nChan = numel(obj.P.chanMap.chanMap);
            else
                nChan = 32; % don't have any other guess
            end
            
            if ~isempty(obj.H.settings.ChooseFileEdt.String) && ...
                    ~strcmp(obj.H.settings.ChooseFileEdt.String, '...')
                b = get_file_size(obj.H.settings.ChooseFileEdt.String);

                a = cast(0, 'int16'); % hard-coded for now, int16
                q = whos('a');
                bytesPerSamp = q.bytes;

                if ~(mod(b,bytesPerSamp)==0 && mod(b/bytesPerSamp,nChan)==0)
                    % try figuring out number of channels, since the previous
                    % guess didn't work
                    testNC = ceil(nChan*0.9):floor(nChan*1.1);
                    possibleVals = testNC(mod(b/bytesPerSamp, testNC)==0);
                    if ~isempty(possibleVals)
                        if ~isempty(find(possibleVals>nChan,1))
                            nChan = possibleVals(find(possibleVals>nChan,1));
                        else
                            nChan = possibleVals(end);
                        end
                        obj.log(sprintf('Guessing that number of channels is %d. If it doesn''t look right, consider trying: %s', nChan, num2str(possibleVals)));
                    else
                        obj.log('Cannot work out a guess for number of channels in the file. Please enter number of channels to proceed.');
                        nChan = [];
                        return;
                    end
                end
                obj.H.settings.setnChanEdt.String = num2str(nChan);
                obj.P.nSamp = b/bytesPerSamp/nChan;
                
                
            end
            
            if nChan~=origNChan
                obj.P.datMMfile = [];
                if nChan>=64
                    obj.P.colormapMode = true;
                    obj.P.nChanToPlotCM = nChan;
                end
            end
        end
        
        function advancedPopup(obj)
            
            % bring up popup window to set other ops
            helpdlg({'To set advanced options, do this in the command window:','',...
                '>> ks = get(gcf, ''UserData'');',...
                '>> ks.ops.myOption = myValue;'})
            
        end
        
        function runAll(obj)
            
            obj.P.ksDone = false;
            
            obj.runPreproc;
            obj.runSpikesort;
            obj.runSaveToPhy;
            
        end
        
        function prepareForRun(obj)
            % TODO check that everything is set up correctly to run
            
            obj.ops.fbinary = obj.H.settings.ChooseFileEdt.String;
            if ~exist(obj.ops.fbinary, 'file')
                obj.log('Cannot run: Data file not found.');
                return;
            end
            
            wd = obj.H.settings.ChooseTempdirEdt.String;
            if ~exist(wd, 'dir')
                obj.log('Cannot run: Working directory not found.');
                return
            end
            obj.ops.fproc = fullfile(wd, 'temp_wh.dat');
            
            obj.ops.saveDir = obj.H.settings.ChooseOutputEdt.String;
            if ~exist(obj.ops.saveDir, 'dir')
                mkdir(obj.ops.saveDir);
            end
            
            % build channel map that includes only the connected channels
            chanMap = struct();
            conn = obj.P.chanMap.connected;
            chanMap.chanMap = obj.P.chanMap.chanMap(conn); 
            chanMap.xcoords = obj.P.chanMap.xcoords(conn); 
            chanMap.ycoords = obj.P.chanMap.ycoords(conn);
            if isfield(obj.P.chanMap, 'kcoords')
                chanMap.kcoords = obj.P.chanMap.kcoords(conn);
            end
            obj.ops.chanMap = chanMap;
            
            % sanitize options set in the gui
            obj.ops.Nfilt = numel(obj.ops.chanMap.chanMap) * 4;
            
            %obj.ops.Nfilt = str2double(obj.H.settings.setNfiltEdt.String);
            if isempty(obj.ops.Nfilt)||isnan(obj.ops.Nfilt)
                obj.ops.Nfilt = numel(obj.ops.chanMap.chanMap)*2.5;
            end
            if mod(obj.ops.Nfilt,32)~=0
                obj.ops.Nfilt = ceil(obj.ops.Nfilt/32)*32;
            end
            %obj.H.settings.setNfiltEdt.String = num2str(obj.ops.Nfilt);
            
            obj.ops.NchanTOT = str2double(obj.H.settings.setnChanEdt.String);
            
            obj.ops.minfr_goodchannels = str2double(obj.H.settings.setMinfrEdt.String);
            if isempty(obj.ops.minfr_goodchannels)||isnan(obj.ops.minfr_goodchannels)
                obj.ops.minfr_goodchannels = 0.1;
            end
            if obj.ops.minfr_goodchannels==0
                obj.ops.throw_out_channels = false;
            else
                obj.ops.throw_out_channels = true;
            end
            obj.H.settings.setMinfrEdt.String = num2str(obj.ops.minfr_goodchannels);

            obj.ops.fs = str2num(obj.H.settings.setFsEdt.String);
            if isempty(obj.ops.fs)||isnan(obj.ops.fs)
                obj.ops.fs = 30000;
            end
                        
            obj.ops.Th = str2num(obj.H.settings.setThEdt.String);
            if isempty(obj.ops.Th)||any(isnan(obj.ops.Th))
                obj.ops.Th = [10 4];
            end
            obj.H.settings.setThEdt.String = num2str(obj.ops.Th);
            
            obj.ops.lam = str2num(obj.H.settings.setLambdaEdt.String);
            if isempty(obj.ops.lam)||isnan(obj.ops.lam)
                obj.ops.lam = 10;
            end
            obj.H.settings.setLambdaEdt.String = num2str(obj.ops.lam);
            
            obj.ops.AUCsplit = str2double(obj.H.settings.setCcsplitEdt.String);
            if isempty(obj.ops.AUCsplit)||isnan(obj.ops.AUCsplit)
                obj.ops.AUCsplit = 0.9;
            end
            obj.H.settings.setCcsplitEdt.String = num2str(obj.ops.AUCsplit);
            
            obj.ops.trange = str2num(obj.H.settings.setTrangeEdt.String);
            if isempty(obj.ops.trange)||any(isnan(obj.ops.trange))
                obj.ops.trange = [0 Inf];
            end
            obj.H.settings.setTrangeEdt.String = num2str(obj.ops.trange);
            
        end
        
        function runPreproc(obj)
            
            obj.prepareForRun;
            
            % do preprocessing
            obj.ops.gui = obj; % for kilosort to access, e.g. calling "log"
            try
                obj.log('Preprocessing...'); 
                obj.rez = preprocessDataSub(obj.ops);
                
                % update connected channels
                igood = obj.rez.ops.igood;
                previousGood = find(obj.P.chanMap.connected);
                newGood = previousGood(igood);
                obj.P.chanMap.connected = false(size(obj.P.chanMap.connected));
                obj.P.chanMap.connected(newGood) = true;
                
                % use the new whitening matrix, which can sometimes be
                % quite different than the earlier estimated one
                rW = obj.rez.Wrot;
                if isfield(obj.P, 'Wrot')                    
                    pW = obj.P.Wrot;                    
                else
                    pW = zeros(numel(obj.P.chanMap.connected));
                end
                cn = obj.P.chanMap.connected;
                pW(cn,cn) = rW;
                obj.P.Wrot = pW;
                
                set(obj.H.settings.runSpikesortBtn, 'enable', 'on');
                
                % update gui with results of preprocessing
                obj.updateDataView();
                obj.log('Done preprocessing.'); 
            catch ex
                obj.log(sprintf('Error preprocessing! %s', ex.message));
                keyboard
            end
            
        end
        
        function computeWhitening(obj)
            obj.log('Computing whitening filter...')
            obj.prepareForRun;
            
            % here, use a different channel map than the actual: show all
            % channels as connected. That way we can drop/add without
            % recomputing everything later. 
            ops = obj.ops;
            ops.chanMap = obj.P.chanMap;
            ops.chanMap.connected = true(size(ops.chanMap.connected));
            
            [~,Wrot] = computeWhitening(ops); % this refers to a function outside the gui
            
            obj.P.Wrot = Wrot;
            obj.updateDataView;
            obj.log('Done.')
        end
        
        function runSpikesort(obj)
            % fit templates
            try
                % pre-clustering to re-order batches by depth
                obj.log('Pre-clustering to re-order batches by depth')
                obj.rez = clusterSingleBatches(obj.rez);
                
                % main optimization
                obj.log('Main optimization')
                obj.rez = learnAndSolve8b(obj.rez);
                
                % final splits and merges
                if 1
                    obj.log('Merges...')
                    obj.rez = find_merges(obj.rez, 1);
                    
                    % final splits by SVD
                    obj.log('Splits part 1/2...')
                    obj.rez = splitAllClusters(obj.rez, 1);
                    
                    % final splits by amplitudes
                    obj.log('Splits part 2/2...')
                    obj.rez = splitAllClusters(obj.rez, 0);
                    
                    % decide on cutoff
                    obj.log('Last step. Setting cutoff...')
                    obj.rez = set_cutoff(obj.rez);
                end
                                                                
                obj.P.ksDone = true;
                
                obj.log('Kilosort finished!');
                set(obj.H.settings.runSaveBtn, 'enable', 'on');
                obj.updateDataView();
            catch ex
                obj.log(sprintf('Error running kilosort! %s', ex.message));
            end   
                        
        end
        
        function runSaveToPhy(obj)            
            % save results
            obj.log(sprintf('Saving data to %s', obj.ops.saveDir));
            % discard features in final rez file (too slow to save)
            rez = obj.rez;
            rez.cProj = [];
            rez.cProjPC = [];
            rez.ops.gui = [];
            
            % save final results as rez2
            fname = fullfile(obj.ops.saveDir, 'rez.mat');
            save(fname, 'rez', '-v7.3');
            
            try
                rezToPhy(obj.rez, obj.ops.saveDir);
            catch ex
                obj.log(sprintf('Error saving data for phy! %s', ex.message));
            end            
            obj.log('Done');
        end
            
        function updateDataView(obj)
            
            if obj.P.dataGood && obj.P.probeGood

                % get currently selected time and channels                
                t = obj.P.currT;
                chList = obj.P.selChans;
                tWin = obj.P.tWin;
                
                % initialize data loading if necessary
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
                    
                    % load and process data
                    datAll = mmf.Data.x(:,samps(1):samps(2));
                    
                    % filtered, whitened
                    obj.prepareForRun();
                    datAllF = ksFilter(datAll, obj.ops);
                    datAllF = double(gather(datAllF));
                    if isfield(obj.P, 'Wrot') && ~isempty(obj.P.Wrot)
                        %Wrot = obj.P.Wrot/obj.ops.scaleproc;
                        conn = obj.P.chanMap.connected;
                        Wrot = obj.P.Wrot(conn,conn);
                        datAllF = datAllF*Wrot;
                    end
                    datAllF = datAllF';
                    
                    if obj.P.ksDone
                        pd = predictData(obj.rez, samps);
                    else
                        pd = zeros(size(datAllF));
                    end
                    
                    dat = datAll(obj.P.chanMap.chanMap(chList),:);
                    
                    connInChList = obj.P.chanMap.connected(chList);
                    
                    cmconn = obj.P.chanMap.chanMap(obj.P.chanMap.connected);
                                        
                    chListW = NaN(size(chList)); % channels within the processed data
                    for q = 1:numel(chList)
                        if obj.P.chanMap.connected(chList(q))
                            chListW(q) = find(cmconn==chList(q),1);
                        end
                    end
                    
                    datW = zeros(numel(chList),size(datAll,2));
                    datP = zeros(numel(chList),size(datAll,2));
                    datW(~isnan(chListW),:) = datAllF(chListW(~isnan(chListW)),:);
                    datP(~isnan(chListW),:) = pd(chListW(~isnan(chListW)),:);
                    datR = datW-datP;
                    
                    if ~obj.P.colormapMode % traces mode
                        
                        ttl = '';
                        
                        
                        
                        if isfield(obj.H, 'dataIm') && ~isempty(obj.H.dataIm)
                            set(obj.H.dataIm, 'Visible', 'off');
                        end
                        
                        if obj.P.showRaw
                            
                            ttl = [ttl '\color[rgb]{0 0 0}raw '];
                            
                            if ~isfield(obj.H, 'dataTr') || numel(obj.H.dataTr)~=numel(chList)
                                % initialize traces
                                if isfield(obj.H, 'dataTr')&&~isempty(obj.H.dataTr); delete(obj.H.dataTr); end
                                obj.H.dataTr = [];
                                hold(obj.H.dataAx, 'on');
                                for q = 1:numel(chList)
                                    obj.H.dataTr(q) = plot(obj.H.dataAx, 0, NaN, 'k', 'LineWidth', 1.5);
                                    set(obj.H.dataTr(q), 'HitTest', 'off');
                                end
                                box(obj.H.dataAx, 'off');
                                %title(obj.H.dataAx, 'scroll and ctrl+scroll to move, alt/shift+scroll to scale/zoom');
                            end                                                

                            conn = obj.P.chanMap.connected(chList);
                            for q = 1:size(dat,1)
                                set(obj.H.dataTr(q), 'XData', (samps(1):samps(2))/Fs, ...
                                    'YData', q+double(dat(q,:)).*obj.P.vScale,...
                                    'Visible', 'on');
                                if conn(q); set(obj.H.dataTr(q), 'Color', 'k');
                                else; set(obj.H.dataTr(q), 'Color', 0.8*[1 1 1]); end
                            end                                                                        
                        elseif isfield(obj.H, 'dataTr')
                            for q = 1:numel(obj.H.dataTr)
                                set(obj.H.dataTr(q), 'Visible', 'off');
                            end
                        end
                        % add filtered, whitened data
                        
                        if obj.P.showWhitened
                            
                            ttl = [ttl '\color[rgb]{0 0.6 0}filtered '];
                            
                            if ~isfield(obj.H, 'ppTr') || numel(obj.H.ppTr)~=numel(chList)
                                % initialize traces
                                if isfield(obj.H, 'ppTr')&&~isempty(obj.H.ppTr); delete(obj.H.ppTr); end
                                obj.H.ppTr = [];
                                hold(obj.H.dataAx, 'on');
                                for q = 1:numel(chList)
                                    obj.H.ppTr(q) = plot(obj.H.dataAx, 0, NaN, 'Color', [0 0.6 0], 'LineWidth', 1.5);
                                    set(obj.H.ppTr(q), 'HitTest', 'off');
                                end
                            end
                            for q = 1:numel(obj.H.ppTr)
                                set(obj.H.ppTr(q), 'Visible', 'on');
                            end

                            for q = 1:numel(chListW)  
                                if ~isnan(chListW(q))
                                    set(obj.H.ppTr(q), 'XData', (samps(1):samps(2))/Fs, ...
                                        'YData', q+datW(q,:).*obj.P.vScale/15);                            
                                end
                            end
                        elseif isfield(obj.H, 'ppTr')
                            for q = 1:numel(obj.H.ppTr)
                                set(obj.H.ppTr(q), 'Visible', 'off');
                            end
                        end
                        
                        if obj.P.showPrediction
                            
                            ttl = [ttl '\color[rgb]{0 0 1}prediction '];
                            
                            if ~isfield(obj.H, 'predTr') || numel(obj.H.predTr)~=numel(chList)
                                % initialize traces
                                if isfield(obj.H, 'predTr')&&~isempty(obj.H.predTr); delete(obj.H.predTr); end
                                obj.H.predTr = [];
                                hold(obj.H.dataAx, 'on');
                                for q = 1:numel(chList)
                                    obj.H.predTr(q) = plot(obj.H.dataAx, 0, NaN, 'b', 'LineWidth', 1.5);
                                    set(obj.H.predTr(q), 'HitTest', 'off');
                                end
                            end
                            for q = 1:numel(obj.H.predTr)
                                set(obj.H.predTr(q), 'Visible', 'on');
                            end

                            for q = 1:numel(chListW)  
                                if ~isnan(chListW(q))
                                    set(obj.H.predTr(q), 'XData', (samps(1):samps(2))/Fs, ...
                                        'YData', q+datP(q,:).*obj.P.vScale/15);                            
                                end
                            end
                        elseif isfield(obj.H, 'predTr')
                            for q = 1:numel(obj.H.predTr)
                                set(obj.H.predTr(q), 'Visible', 'off');
                            end
                        end
                        
                        if obj.P.showResidual
                            
                            ttl = [ttl '\color[rgb]{1 0 0}residual '];
                            
                            if ~isfield(obj.H, 'residTr') || numel(obj.H.residTr)~=numel(chList)
                                % initialize traces
                                if isfield(obj.H, 'residTr')&&~isempty(obj.H.residTr); delete(obj.H.residTr); end
                                obj.H.residTr = [];
                                hold(obj.H.dataAx, 'on');
                                for q = 1:numel(chList)
                                    obj.H.residTr(q) = plot(obj.H.dataAx, 0, NaN, 'r', 'LineWidth', 1.5);
                                    set(obj.H.residTr(q), 'HitTest', 'off');
                                end
                            end
                            for q = 1:numel(obj.H.residTr)
                                set(obj.H.residTr(q), 'Visible', 'on');
                            end

                            for q = 1:numel(chListW)  
                                if ~isnan(chListW(q))
                                    set(obj.H.residTr(q), 'XData', (samps(1):samps(2))/Fs, ...
                                        'YData', q+datR(q,:).*obj.P.vScale/15);                            
                                end
                            end
                        elseif isfield(obj.H, 'residTr')
                            for q = 1:numel(obj.H.residTr)
                                set(obj.H.residTr(q), 'Visible', 'off');
                            end
                        end
                            
                        title(obj.H.dataAx, ttl);
                        yt = arrayfun(@(x)sprintf('%d (%d)', chList(x), obj.P.chanMap.chanMap(chList(x))), 1:numel(chList), 'uni', false);
                        set(obj.H.dataAx, 'YTick', 1:numel(chList), 'YTickLabel', yt);
                        set(obj.H.dataAx, 'YLim', [0 numel(chList)+1], 'YDir', 'normal');
                    else % colormap mode
                        %chList = 1:numel(obj.P.chanMap.chanMap);
                        
                        if ~isfield(obj.H, 'dataIm') || isempty(obj.H.dataIm)
                            obj.H.dataIm = [];
                            hold(obj.H.dataAx, 'on');
                            obj.H.dataIm = imagesc(obj.H.dataAx, chList, ...
                                (samps(1):samps(2))/Fs,...
                                datAll(obj.P.chanMap.connected,:));
                            set(obj.H.dataIm, 'HitTest', 'off');                            
                            colormap(obj.H.dataAx, obj.P.colormap);
                        end
                        
                        if isfield(obj.H, 'dataTr') && ~isempty(obj.H.dataTr)
                            for q = 1:numel(obj.H.dataTr)
                                set(obj.H.dataTr(q), 'Visible', 'off');
                            end
                        end
                        if isfield(obj.H, 'ppTr') && ~isempty(obj.H.ppTr)
                            for q = 1:numel(obj.H.ppTr)
                                set(obj.H.ppTr(q), 'Visible', 'off');
                            end
                        end
                        if isfield(obj.H, 'predTr') && ~isempty(obj.H.predTr)
                            for q = 1:numel(obj.H.predTr)
                                set(obj.H.predTr(q), 'Visible', 'off');
                            end
                        end
                        if isfield(obj.H, 'residTr') && ~isempty(obj.H.residTr)
                            for q = 1:numel(obj.H.residTr)
                                set(obj.H.residTr(q), 'Visible', 'off');
                            end
                        end
                        
                        set(obj.H.dataIm, 'Visible', 'on');
                        if obj.P.showRaw
                            set(obj.H.dataIm, 'XData', (samps(1):samps(2))/Fs, ...
                                'YData', 1:sum(connInChList), 'CData', dat(connInChList,:));                    
                            set(obj.H.dataAx, 'CLim', [-1 1]*obj.P.vScale*15000);
                            title(obj.H.dataAx, 'raw');
                        elseif obj.P.showWhitened
                            set(obj.H.dataIm, 'XData', (samps(1):samps(2))/Fs, ...
                                'YData', 1:sum(connInChList), 'CData', datW(connInChList,:));                    
                            set(obj.H.dataAx, 'CLim', [-1 1]*obj.P.vScale*225000);
                            title(obj.H.dataAx, 'filtered');
                        elseif obj.P.showPrediction
                            set(obj.H.dataIm, 'XData', (samps(1):samps(2))/Fs, ...
                                'YData', 1:sum(connInChList), 'CData', datP(connInChList,:));                    
                            set(obj.H.dataAx, 'CLim', [-1 1]*obj.P.vScale*225000);
                            title(obj.H.dataAx, 'prediction');
                        else % obj.P.showResidual
                            set(obj.H.dataIm, 'XData', (samps(1):samps(2))/Fs, ...
                                'YData', 1:sum(connInChList), 'CData', datR(connInChList,:));                    
                            set(obj.H.dataAx, 'CLim', [-1 1]*obj.P.vScale*225000);
                            title(obj.H.dataAx, 'residual');
                        end
                        set(obj.H.dataAx, 'YLim', [0 sum(connInChList)], 'YDir', 'normal');
                    end
                    
                    set(obj.H.dataAx, 'XLim', t+tWin);
                    
                    
                    set(obj.H.dataAx, 'YTickLabel', []);
                end
                
            end
            
            
            
        end
        
        
        function updateProbeView(obj, varargin)
            
            if ~isempty(varargin) % any argument means to re-initialize
            
                obj.P.probeGood = false; % might have just selected a new one

                % if probe file exists, load it
                selProbe = obj.H.settings.setProbeEdt.String{obj.H.settings.setProbeEdt.Value};

                cm = [];
                switch selProbe
                    case '[new]'
                        %obj.log('New probe creator not yet implemented.');
                        answer = inputdlg({'Name for new channel map:', ...
                            'X-coordinates of each site (can use matlab expressions):',...
                            'Y-coordinates of each site:',...
                            'Shank index (''kcoords'') for each site (blank for single shank):',...
                            'Channel map (the list of rows in the data file for each site):',...
                            'List of disconnected/bad site numbers (blank for none):'});
                        if isempty(answer)
                            return;
                        else
                            cm.name = answer{1};
                            cm.xcoords = str2num(answer{2});
                            cm.ycoords = str2num(answer{3});
                            if ~isempty(answer{4})
                                cm.kcoords = str2num(answer{4});
                            end
                            cm.chanMap = str2num(answer{5});
                            if ~isempty(answer{6})
                                q = str2num(answer{6});
                                if numel(q) == numel(cm.chanMap)
                                    cm.connected = q;
                                else
                                    cm.connected = true(size(cm.chanMap));
                                    cm.connected(q) = false;
                                end
                            end
                            cm = createValidChanMap(cm);
                            if ~isempty(cm)
                                obj.P.allChanMaps(end+1) = cm;
                                currProbeList = obj.H.settings.setProbeEdt.String;
                                newProbeList = [{cm.name}; currProbeList];
                                obj.H.settings.setProbeEdt.String = newProbeList;
                                obj.H.settings.setProbeEdt.Value = 1;
                                answer = questdlg('Save this channel map for later?');
                                if strcmp(answer, 'Yes')
                                    saveNewChanMap(cm, obj);
                                end
                            else
                                obj.log('Channel map invalid. Must have chanMap, xcoords, and ycoords of same length');
                                return;
                            end
                        end
                    case 'other...'
                        [filename, pathname] = uigetfile('*.mat', 'Pick a channel map file.');
                        
                        if filename~=0 % 0 when cancel
                            %obj.log('choosing a different channel map not yet implemented.');
                            cm = load(fullfile(pathname, filename));
                            cm = createValidChanMap(cm, filename);
                            if ~isempty(cm)
                                obj.P.allChanMaps(end+1) = cm;
                                currProbeList = obj.H.settings.setProbeEdt.String;
                                newProbeList = [{cm.name}; currProbeList];
                                obj.H.settings.setProbeEdt.String = newProbeList;
                                obj.H.settings.setProbeEdt.Value = 1;
                                answer = questdlg('Save this channel map for later?');
                                if strcmp(answer, 'Yes')
                                    saveNewChanMap(cm, obj);
                                end
                            else
                                obj.log('Channel map invalid. Must have chanMap, xcoords, and ycoords of same length');
                                return;
                            end
                        else
                            return;
                        end
                    otherwise
                        probeNames = {obj.P.allChanMaps.name};
                        cm = obj.P.allChanMaps(strcmp(probeNames, selProbe));
                end               
                
                nSites = numel(cm.chanMap);
                ux = unique(cm.xcoords); uy = unique(cm.ycoords);
                
                if isfield(cm, 'siteSize') && ~isempty(cm.siteSize)
                    ss = cm.siteSize(1);
                else
                    ss = min(diff(uy));
                end
                cm.siteSize = ss;
                               
                if isfield(obj.H, 'probeSites')&&~isempty(obj.H.probeSites)
                    delete(obj.H.probeSites);
                    obj.H.probeSites = [];
                end
                
                obj.P.chanMap = cm;
                obj.P.probeGood = true;
                
                nChan = checkNChan(obj);
                
                if obj.P.dataGood
                    obj.computeWhitening()

                    set(obj.H.settings.runBtn, 'enable', 'on');
                    set(obj.H.settings.runPreprocBtn, 'enable', 'on');
                end
            end
            
            if obj.P.probeGood
                cm = obj.P.chanMap;
                
                if ~isempty(cm)
                    % if it is valid, plot it
                    nSites = numel(cm.chanMap);
                    ss = cm.siteSize;
                    
                    if ~isfield(obj.H, 'probeSites') || isempty(obj.H.probeSites) || ...
                            numel(obj.H.probeSites)~=nSites 
                        obj.H.probeSites = [];
                        hold(obj.H.probeAx, 'on');                    
                        sq = ss*([0 0; 0 1; 1 1; 1 0]-[0.5 0.5]);
                        for q = 1:nSites
                            obj.H.probeSites(q) = fill(obj.H.probeAx, ...
                                sq(:,1)+cm.xcoords(q), ...
                                sq(:,2)+cm.ycoords(q), 'b');
                            set(obj.H.probeSites(q), 'HitTest', 'off');
                                                        
                        end
                        yc = cm.ycoords;
                        ylim(obj.H.probeAx, [min(yc) max(yc)]);
                        axis(obj.H.probeAx, 'equal');
                        set(obj.H.probeAx, 'XTick', [], 'YTick', []);
                        title(obj.H.probeAx, {'scroll to zoom, click to view channel,', 'right-click to disable channel'});
                        %axis(obj.H.probeAx, 'off');
                    end

                    y = obj.P.currY;
                    x = obj.P.currX;
                    if obj.P.colormapMode
                        nCh = obj.P.nChanToPlotCM;
                        if nCh>numel(cm.chanMap)
                            nCh = numel(cm.chanMap);
                        end
                    else
                        nCh = obj.P.nChanToPlot;
                    end
                    conn = obj.P.chanMap.connected;

                    dists = ((cm.xcoords-x).^2 + (cm.ycoords-y).^2).^(0.5);
                    [~, ii] = sort(dists);
                    obj.P.selChans = ii(1:nCh);

                    % order by y-coord
                    yc = obj.P.chanMap.ycoords;
                    theseYC = yc(obj.P.selChans);
                    [~,ii] = sort(theseYC);
                    obj.P.selChans = obj.P.selChans(ii);
                    
                    for q = 1:nSites
                        if ismember(q, obj.P.selChans) && ~conn(q)
                            set(obj.H.probeSites(q), 'FaceColor', [1 1 0]);
                        elseif ismember(q, obj.P.selChans) 
                            set(obj.H.probeSites(q), 'FaceColor', [0 1 0]);
                        elseif ~conn(q)
                            set(obj.H.probeSites(q), 'FaceColor', [1 0 0]);
                        else
                            set(obj.H.probeSites(q), 'FaceColor', [0 0 1]);
                        end
                    end
                    
                end
            end
        end
        
        function scrollCB(obj,~,evt)
            
            if obj.isInLims(obj.H.dataAx)
                % in data axis                
                if obj.P.dataGood
                    m = get(obj.H.fig,'CurrentModifier');

                    if isempty(m)  
                        % scroll in time
                        maxT = obj.P.nSamp/obj.ops.fs;
                        winSize = diff(obj.P.tWin);
                        shiftSize = -evt.VerticalScrollCount*winSize*0.1;
                        obj.P.currT = obj.P.currT+shiftSize;
                        if obj.P.currT>maxT; obj.P.currT = maxT; end
                        if obj.P.currT<0; obj.P.currT = 0; end
                         
                    elseif strcmp(m, 'shift')
                        % zoom in time
                        maxT = obj.P.nSamp/obj.ops.fs;
                        oldWin = obj.P.tWin+obj.P.currT;
                        newWin = ksGUI.chooseNewRange(oldWin, ...
                            1.2^evt.VerticalScrollCount,...
                            diff(oldWin)/2+oldWin(1), [0 maxT]);
                        obj.P.tWin = newWin-newWin(1);
                        obj.P.currT = newWin(1);
                        
                    elseif strcmp(m, 'control')
                        if obj.P.colormapMode
                            % zoom in Y when in colormap mode
                            obj.P.nChanToPlotCM = round(obj.P.nChanToPlotCM*1.2^evt.VerticalScrollCount);
                            if obj.P.nChanToPlotCM>numel(obj.P.chanMap.chanMap)
                                obj.P.nChanToPlotCM=numel(obj.P.chanMap.chanMap);
                            elseif obj.P.nChanToPlotCM<1
                                obj.P.nChanToPlotCM=1;
                            end
                        else
                            % scroll in channels when in traceview
                            obj.P.currY = obj.P.currY-evt.VerticalScrollCount*...
                                min(diff(unique(obj.P.chanMap.ycoords)));
                            yc = obj.P.chanMap.ycoords;
                            mx = max(yc)+obj.P.chanMap.siteSize;
                            mn = min(yc)-obj.P.chanMap.siteSize;
                            if obj.P.currY>mx; obj.P.currY = mx; end
                            if obj.P.currY<mn; obj.P.currY = mn; end
                        end
                        obj.updateProbeView();
                        
                        
                    elseif strcmp(m, 'alt')
                        % zoom in scaling of traces
                        obj.P.vScale = obj.P.vScale*1.2^(-evt.VerticalScrollCount);
                        
                    end
                    obj.updateDataView();
                end
            elseif obj.isInLims(obj.H.probeAx)
                % in probe axis
                if obj.P.probeGood
                    cpP = get(obj.H.probeAx, 'CurrentPoint');
                    yl = get(obj.H.probeAx, 'YLim');
                    currY = cpP(1,2);
                    yc = obj.P.chanMap.ycoords;
                    mx = max(yc)+obj.P.chanMap.siteSize;
                    mn = min(yc)-obj.P.chanMap.siteSize;
                    newyl = ksGUI.chooseNewRange(yl, ...
                        1.2^evt.VerticalScrollCount,...
                        currY, [mn mx]);          
                    set(obj.H.probeAx, 'YLim', newyl);
                end
            end
        end
        
        function probeClickCB(obj, ~, keydata)
            if keydata.Button==1 % left click
                obj.P.currX = round(keydata.IntersectionPoint(1));
                obj.P.currY = round(keydata.IntersectionPoint(2));
            else % any other click, disconnect/reconnect the nearest channel
                thisX = round(keydata.IntersectionPoint(1));
                thisY = round(keydata.IntersectionPoint(2));
                xc = obj.P.chanMap.xcoords;
                yc = obj.P.chanMap.ycoords;
                dists = ((thisX-xc).^2+(thisY-yc).^2).^(0.5);
                [~,ii] = sort(dists);
                obj.P.chanMap.connected(ii(1)) = ~obj.P.chanMap.connected(ii(1));                
            end
            obj.updateProbeView;
            obj.updateDataView;
        end
        
        function dataClickCB(obj, ~, keydata)                   
            if keydata.Button==1 % left click, re-center view
                obj.P.currT = keydata.IntersectionPoint(1)-diff(obj.P.tWin)/2;
                
                thisY = round(keydata.IntersectionPoint(2)); 
                if thisY<=0; thisY = 1; end
                if thisY>=numel(obj.P.selChans); thisY = numel(obj.P.selChans); end
                thisCh = obj.P.selChans(thisY);
                obj.P.currY = obj.P.chanMap.ycoords(thisCh);
                
            else % any other click, disconnect/reconnect the nearest channel   
                thisY = round(keydata.IntersectionPoint(2));  
                if thisY<=0; thisY = 1; end
                if obj.P.colormapMode
                    sc = obj.P.selChans;
                    cm = obj.P.chanMap;
                    sc = sc(ismember(sc, find(cm.connected)));
                    if thisY<=0; thisY = 1; end
                    if thisY>=numel(sc); thisY = numel(sc); end
                    thisCh = sc(thisY);                    
                else
                    if thisY>=numel(obj.P.selChans); thisY = numel(obj.P.selChans); end
                    thisCh = obj.P.selChans(thisY);                
                end
                obj.P.chanMap.connected(thisCh) = ~obj.P.chanMap.connected(thisCh);
            end
            obj.updateProbeView;
            obj.updateDataView;
        end
        
        function timeClickCB(obj, ~, keydata)
            if obj.P.dataGood
                nSamp = obj.P.nSamp;
                maxT = nSamp/obj.ops.fs;
                
                obj.P.currT = keydata.IntersectionPoint(1)*maxT;
                set(obj.H.timeLine, 'XData', keydata.IntersectionPoint(1)*[1 1]);
                
                obj.updateDataView;
            end
            
        end
        
        function keyboardFcn(obj, ~, k)
            switch k.Key
                case 'uparrow'
                    obj.P.nChanToPlot = obj.P.nChanToPlot+1;
                    if obj.P.nChanToPlot > numel(obj.P.chanMap.chanMap)
                        obj.P.nChanToPlot = numel(obj.P.chanMap.chanMap);
                    end
                case 'downarrow'
                    obj.P.nChanToPlot = obj.P.nChanToPlot-1;
                    if obj.P.nChanToPlot == 0
                        obj.P.nChanToPlot = 1;
                    end
                case 'c'
                    obj.P.colormapMode = ~obj.P.colormapMode;     
                case '1'
                    if obj.P.colormapMode 
                        obj.P.showRaw = true; 
                        obj.P.showWhitened = false;
                        obj.P.showPrediction = false;
                        obj.P.showResidual = false;
                    else
                        obj.P.showRaw = ~obj.P.showRaw;
                    end
                case '2'
                    if obj.P.colormapMode
                        obj.P.showRaw = false;
                        obj.P.showWhitened = true;
                        obj.P.showPrediction = false;
                        obj.P.showResidual = false;
                    else
                        obj.P.showWhitened = ~obj.P.showWhitened;
                    end
                case '3'
                    if obj.P.colormapMode
                        obj.P.showRaw = false;
                        obj.P.showWhitened = false;
                        obj.P.showPrediction = true;
                        obj.P.showResidual = false;
                    else
                        obj.P.showPrediction = ~obj.P.showPrediction;
                    end
                case '4'
                    if obj.P.colormapMode
                        obj.P.showRaw = false;
                        obj.P.showWhitened = false;
                        obj.P.showPrediction = false;
                        obj.P.showResidual = true;
                    else
                        obj.P.showResidual = ~obj.P.showResidual;
                    end
                    
            end
            obj.updateProbeView;
            obj.updateDataView;
        end
            
        
        function saveGUIsettings(obj)
            
            saveDat.settings.ChooseFileEdt.String = obj.H.settings.ChooseFileEdt.String;
            saveDat.settings.ChooseTempdirEdt.String = obj.H.settings.ChooseTempdirEdt.String;
            saveDat.settings.setProbeEdt.String = obj.H.settings.setProbeEdt.String;
            saveDat.settings.setProbeEdt.Value = obj.H.settings.setProbeEdt.Value;
            saveDat.settings.setnChanEdt.String = obj.H.settings.setnChanEdt.String;
            saveDat.settings.setFsEdt.String = obj.H.settings.setFsEdt.String;
            saveDat.settings.setThEdt.String = obj.H.settings.setThEdt.String;
            saveDat.settings.setLambdaEdt.String = obj.H.settings.setLambdaEdt.String;
            saveDat.settings.setCcsplitEdt.String = obj.H.settings.setCcsplitEdt.String;
            saveDat.settings.setMinfrEdt.String = obj.H.settings.setMinfrEdt.String;
            
            saveDat.ops = obj.ops;
            saveDat.ops.gui = [];
            saveDat.rez = obj.rez;
            saveDat.rez.cProjPC = []; 
            saveDat.rez.cProj = []; 
            saveDat.rez.ops.gui = [];
            saveDat.P = obj.P;
            
            if ~strcmp(obj.H.settings.ChooseFileEdt.String, '...')
                [p,fn] = fileparts(obj.H.settings.ChooseFileEdt.String);            
                savePath = fullfile(p, [fn '_ksSettings.mat']);
                save(savePath, 'saveDat', '-v7.3');
            end            
            
            %obj.refocus(obj.H.settings.saveBtn);
        end
        
        function restoreGUIsettings(obj)
            [p,fn] = fileparts(obj.H.settings.ChooseFileEdt.String);            
            savePath = fullfile(p, [fn '_ksSettings.mat']);
            
            if exist(savePath, 'file')
                obj.log('Restoring saved session...');
                
                load(savePath);
                
                obj.H.settings.ChooseFileEdt.String = saveDat.settings.ChooseFileEdt.String;
                obj.H.settings.ChooseTempdirEdt.String = saveDat.settings.ChooseTempdirEdt.String;
                obj.H.settings.setProbeEdt.String = saveDat.settings.setProbeEdt.String;
                obj.H.settings.setProbeEdt.Value = saveDat.settings.setProbeEdt.Value;
                obj.H.settings.setnChanEdt.String = saveDat.settings.setnChanEdt.String;
                obj.H.settings.setFsEdt.String = saveDat.settings.setFsEdt.String;
                obj.H.settings.setThEdt.String = saveDat.settings.setThEdt.String;
                obj.H.settings.setLambdaEdt.String = saveDat.settings.setLambdaEdt.String;
                obj.H.settings.setCcsplitEdt.String = saveDat.settings.setCcsplitEdt.String;
                obj.H.settings.setMinfrEdt.String = saveDat.settings.setMinfrEdt.String;
                
                obj.ops = saveDat.ops;
                obj.rez = saveDat.rez;
                obj.P = saveDat.P;
                
                obj.updateProbeView('new');
                obj.updateDataView;
            end
        end
            
        function writeScript(obj)
            % write a .m file script that the user can use later to run
            % directly, i.e. skipping the gui
            obj.log('Writing to script not yet implemented.');
            obj.refocus(obj.H.settings.writeBtn);
        end
        
        function help(obj)
            
            hstr = {'Welcome to Kilosort!',...                
                '',...
                '*** Troubleshooting ***', ...
                '1. Click ''reset'' to try to clear any GUI problems or weird errors. Also try restarting matlab.', ...                
                '2. Visit github.com/MouseLand/Kilosort2 to see more troubleshooting tips.',...
                '3. Create an issue at github.com/MouseLand/Kilosort2 with as much detail about the problem as possible.'};
            
            h = helpdlg(hstr, 'Kilosort help');
            
        end
        
        function reset(obj)
             % full reset: delete userSettings.mat and the settings file
             % for current file. re-launch. 
             
             if exist(obj.P.settingsPath)
                delete(obj.P.settingsPath);
             end
                
             [p,fn] = fileparts(obj.H.settings.ChooseFileEdt.String);
             savePath = fullfile(p, [fn '_ksSettings.mat']);
             if exist(savePath)
                delete(savePath);
             end
             
             obj.P.skipSave = true;
             kilosort;
             
        end
        
        function cleanup(obj)
            if ~isfield(obj.P, 'skipSave')
                obj.saveGUIsettings();
            end
            fclose('all');
        end
        
        function log(obj, message)
            % show a message to the user in the log box
            timestamp = datestr(now, 'dd-mm-yyyy HH:MM:SS');
            str = sprintf('[%s] %s', timestamp, message);
            current = get(obj.H.logBox, 'String');
            set(obj.H.logBox, 'String', [current; str], ...
                'Value', numel(current) + 1);
            drawnow;
        end
    end
    
    methods(Static)
        
        function refocus(uiObj)
            set(uiObj, 'Enable', 'off');
            drawnow update;
            set(uiObj, 'Enable', 'on');
        end
        
        function ops = defaultOps()
            % look for a default ops file and load it
%             if exist('defaultOps.mat')
%                 load('defaultOps.mat', 'ops');
            if exist('configFile384.m', 'file')
                configFile384;  
                ops.trange      = [0 Inf];
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
        
        function newRange = chooseNewRange(oldRange, scaleFactor,newCenter, maxRange)
            dRange = diff(oldRange);
            mn = maxRange(1); mx = maxRange(2);
            
            if newCenter>mx; newCenter = mx; end
            if newCenter<mn; newCenter = mn; end
            dRange = dRange*scaleFactor;
            
            if dRange>(mx-mn); dRange = mx-mn; end
            newRange = newCenter+[-0.5 0.5]*dRange;
            if newRange(1)<mn
                newRange = newRange+mn-newRange(1);
            elseif newRange(2)>mx
                newRange = newRange+mx-newRange(2);
            end
            
        end
        
    end
    
end


