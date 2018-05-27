

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

    properties
        figHandle % handle to the figure
        
        guiHandles % struct of handles to useful parts of the gui
        
        ops % struct for kilosort to run
        
        rez % struct of results of running
        
    end
    
    methods
        function obj = ksGUI(parent)
            
            obj.init();
            
            obj.build(parent);
            
        end
        
        function init(obj)
            
            % check that required functions are present
            if ~exist('uiextras.HBox')
                error('ksGUI:init:uix', 'You must have the "uiextras" toolbox to use this GUI. Choose Environment->Get Add-ons and search for "GUI Layout Toolbox" by David Sampson.\n')
            end
            
            % add paths
            mfPath = mfilename('fullpath');
            if ~exist('preprocessDataSub')
                addpath(genpath(fileparts(mfPath)));
            end
            if ~exist('readNPY')
                githubDir = fileparts(fileparts(mfPath)); % taking a guess that they have a directory with all github repos
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
                    cd(fullfile(fileparts(mfPath), 'CUDA'));
                    mexGPUall;
                    fprintf(1, 'Success!\n');
                    cd(oldDir);
                catch ex
                    fprintf(1, 'Compilation failed. Check installation instructions at https://github.com/cortex-lab/Kilosort\n');
                    rethrow(ex);
                end
            end
            
            obj.ops = ksGUI.defaultOps();
            
        end
        
        
        function build(obj, f)
            % construct the GUI with appropriate panels
            obj.figHandle = f;
            
            obj.guiHandles.root = uiextras.VBox('Parent', f,...
                'DeleteFcn', @(~,~)obj.cleanup(), 'Visible', 'on', ...
                'Padding', 5);
            
            % - Root sections
            obj.guiHandles.titleBar = uicontrol(...
                'Parent', obj.guiHandles.root,...
                'Style', 'text', 'HorizontalAlignment', 'left', ...
                'String', 'Kilosort', 'FontSize', 36,...
                'FontName', 'Myriad Pro', 'FontWeight', 'bold');
            
            obj.guiHandles.mainSection = uiextras.HBox(...
                'Parent', obj.guiHandles.root);
            
            obj.guiHandles.logPanel = uiextras.Panel(...
                'Parent', obj.guiHandles.root, ...
                'Title', 'Message Log', 'FontSize', 18,...
                'FontName', 'Myriad Pro');
            
            obj.guiHandles.root.Sizes = [-1 -8 -2];
            
            % -- Main section
            obj.guiHandles.settingsPanel = uiextras.Panel(...
                'Parent', obj.guiHandles.mainSection, ...
                'Title', 'Settings', 'FontSize', 18,...
                'FontName', 'Myriad Pro');
            
            obj.guiHandles.probePanel = uiextras.Panel(...
                'Parent', obj.guiHandles.mainSection, ...
                'Title', 'Probe view', 'FontSize', 18,...
                'FontName', 'Myriad Pro');
            
            obj.guiHandles.dataPanel = uiextras.Panel(...
                'Parent', obj.guiHandles.mainSection, ...
                'Title', 'Data view', 'FontSize', 18,...
                'FontName', 'Myriad Pro');
            
            obj.guiHandles.mainSection.Sizes = [-1 -1 -2];
            
            % --- Settings panel
            obj.guiHandles.settingsVBox = uiextras.VBox(...
                'Parent', obj.guiHandles.settingsPanel);
            
            obj.guiHandles.settingsGrid = uiextras.Grid(...
                'Parent', obj.guiHandles.settingsVBox);
            
            % choose file
            obj.guiHandles.settings.ChooseFileTxt = uicontrol(...
                'Parent', obj.guiHandles.settingsGrid,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Select data file');
            obj.guiHandles.settings.ChooseFileEdt = uicontrol(...
                'Parent', obj.guiHandles.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '...');
            
            % choose temporary directory
            
            % choose output path
            
            % set nChannels
            
            % set Fs
            
            % choose probe
            
            % choose max number of clusters
            
            set( obj.guiHandles.settingsGrid, ...
                'ColumnSizes', [-1 -1], 'RowSizes', -1 );%-1*ones(1,7)
            
            
            % button for advanced options
            
            % button for run
            
            % button for write script
            
            % button for save defaults
            
            % -- Message log
            obj.guiHandles.logBox = uicontrol(...
                'Parent', obj.guiHandles.logPanel,...
                'Style', 'listbox', 'Enable', 'inactive', 'String', {}, ...
                'Tag', 'Logging Display', 'FontSize', 14);
            
            obj.log('Initialization success.');
            
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
            
            % get currently selected time and channels
            
            % if there is a file and enough properties to load it, show raw
            % data traces
            
            % if the preprocessing is complete, add whitened data
            
            % if kilosort is finished running, add residuals
            
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
            current = get(obj.guiHandles.logBox, 'String');
            set(obj.guiHandles.logBox, 'String', [current; str], ...
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
        
    end
    
end


