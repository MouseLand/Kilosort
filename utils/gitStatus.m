function ops = gitStatus(ops)
% function ops = gitStatus(ops)
% 
% Create all git tracking status & info necessary to fully reconstitute source code used
% during the experiment
% - enable for use by setting flag:
%       [ops.useGit] == true;
% - if enabled, will track kilosort git repo by default
% 
% Each fieldname under [ops.git] should correspond to a separate git repo to be tracked
% - [ops.git.kilosort]  repo will be included automatically
% 
% - additional repos can be added by including repo name & main function names to ops.git:
%       ops.git.<myRepo>.mainFxn = 'aUniqueFilename.m';
%       e.g.  ops.git.kilosort.mainFxn = 'kilosort.m';
% - if [.mainFxn] does not exist or isempty, will attempt to use the repo name itself:  what(<myRepo>)
% - for additional info, see comments in code
% 
% ---
% [REVIVAL EXAMPLE]
% To revive a git repo to the source/state of Kilosort [rez] struct:
% 
%     % load a saved [rez] struct data file [.PDS]
%     thisFile = fullfile( myDataPath, 'rez.mat');
%     load(thisFile);
%     ops = rez.ops;
% 
%     % create a unique destination for the new repo
%     [~,origDest] = fileparts(ops.saveDir);
%     myNewRepo = fullfile(pwd, sprintf('ksRepo_%s', origDest));
% 
%     % clone a copy from the repo origin
%     cloneRepoStr = sprintf('git clone %s %s', ops.git.kilosort.remote.origin, myNewRepo);
%     [err, resp] = system( cloneRepoStr )
% 
%     % checkout the appropriate commit revision
%     checkoutStr = sprintf('git -C %s checkout %s', myNewRepo, ops.git.kilosort.revision);
%     [err, resp] = system( checkoutStr )
% 
%     % create a patch file from the git diff at the time of Kilosort execution
%     if ~isempty(ops.git.kilosort.diff)
%         fid = fopen('mydiff.patch','w+');
%         fwrite(fid, ops.git.kilosort.diff),
%         fwrite(fid, sprintf('\n')), % ensure trailing newline character
%         fclose(fid);
%         % apply patch to your new repo
%         [err, resp] = system(sprintf('git -C %s apply --whitespace=nowarn %s', myNewRepo, '../mydiff.patch'))
%     end
% 
%     fprintf('Done!\nNew repo located in:\n\t%s\n', myNewRepo);
% ---
%
% 2021-03-08  TBC  Wrote it.    (T.Czuba; czuba@utexas.edu)
% 2021-04-06  TBC  Revised to include full recovery capibility, w/example
%                  Appended git() dependency as subfunction
% 

% NOTE:  Uses lightweight git() wrapper function (**now appended as subfunction**)
%   Direct system() calls, as in revival example, are sufficient on unix-based OSes (Linux, MacOS).
%   If subfunction is necessary on other systems, consider pulling out code for your own
%   external copy. --TBC 2021
% 


if getOr(ops, 'useGit', 1)
    
    if ~isfield(ops,'git') || ~isfield(ops.git,'kilosort')
        % initialize
        ops.git.kilosort = struct('mainFxn','kilosort.m');
    end
    
    % find other git repos that should be tracked
    fn = fieldnames(ops.git);
    fn = ['kilosort', fn(~strcmp(fn, 'kilosort'))];
    
    % Each fieldname under [ops.git] should correspond to a separate git repo to be tracked
    % - [.kilosort]  will be tracked by default
    % - follow format to add additional repos (e.g. lab-specific kilosort config, data preprocessing/conversion, utilities, etc)

    % Attempt git tracking on each repo
    for i = 1:length(fn)
        try
            % only execute git check once
            if ~isfield(ops.git.(fn{i}),'status') || isempty(ops.git.(fn{i}).status)
                
                % determine [basePath] of repo
                if ~isfield(ops.git.(fn{i}), 'basePath') || isempty(ops.git.(fn{i}).basePath)
                    % Try using [mainFxn] location in matlab path first
                    if ~isfield(ops.git.(fn{i}), 'mainFxn') || isempty(ops.git.(fn{i}).mainFxn)
                        % if no [mainFxn] provided, assume repo name matches a directory in matlab path'
                        ops.git.(fn{i}).mainFxn = [];
                        w = what(fn{i});
                        if ~isempty(w.path)
                            ops.git.(fn{i}).basePath = w(1).path;
                        else
                            ops.git.(fn{i}).basePath = []; % still not found
                        end
                    else
                        % find current instance of [mainFxn] in path
                        ops.git.(fn{i}).basePath = fileparts( which(ops.git.(fn{i}).mainFxn) );
                    end
                    
                    % unwrap any enclosing matlab "@" class in the .basePath
                    while contains(ops.git.(fn{i}).basePath, '@')
                        ops.git.(fn{i}).basePath = fileparts(ops.git.(fn{i}).basePath);
                    end
                end
                thisBase = ops.git.(fn{i}).basePath; % shorthand
                
                ops.git.(fn{i}).status          = git(['-C ' thisBase ' status']);
                ops.git.(fn{i}).remote.origin   = git(['-C ' thisBase ' remote get-url origin']);
                ops.git.(fn{i}).branch          = git(['-C ' thisBase ' symbolic-ref --short HEAD']);
                ops.git.(fn{i}).revision        = git(['-C ' thisBase ' rev-parse HEAD']);
                ops.git.(fn{i}).diff            = git(['-C ' thisBase ' diff']);
            end
        catch
            % separate error field allows everything up to error to carry through
            errString = sprintf('%s repo not found, or inaccessible.',fn{i})
            ops.git.(fn{i}).error      = errString;
            % keyboard
        end
    end
end

end % main function


% % % % % % % % % %
%% Sub-functions
% % % % % % % % % %


%% git
function [result, err] = git(varargin)
% function [result, err] = git(varargin)
%
% A thin MATLAB wrapper for Git.
%   Short instructions:
%       Use this exactly as you would use the OS command-line verison of Git.
%
%       This is not meant to be a comprehensive guide to the near-omnipotent
%       Git SCM:
%           http://git-scm.com/documentation
%
%   Useful resources:
%       1. GitX: A visual interface for Git on the OS X client
%       2. Github.com: Remote hosting for Git repos
%       3. Git on Wikipedia: Further reading
%
% v0.1,     27 October 2010 -- MR: Initial support for OS X & Linux,
%                               untested on PCs, but expected to work
% v0.2,     11 March 2011   -- TH: Support for PCs
% v0.3,     12 March 2011   -- MR: Fixed man pages hang bug using redirection
% v0.4,     20 November 2013-- TN: Searching for git in default directories,
%                               returning results as variable
% 2021-04-06  TBC   include optional [err] status output
%                   trim trailing newline char by default
%
% Contributors: (MR) Manu Raghavan
%               (TH) Timothy Hansell
%               (TN) Tassos Natsakis
%

gitlocation='git ';

% Test to see if git is installed
[err, ~] = system([gitlocation '--version']);
% if git is in the path this will return a status of 0
% it will return a 1 only if the command is not found
if err
    gitlocation = [GetGitPath gitlocation];
    [err, ~] = system([gitlocation '--version']);
end

if err
    % Checking if git exists in the default installation folders (for
    % Windows)
    if ispc
        search = system('dir /s /b "c:\Program Files\Git\bin\git.exe');
        searchx86 = system('dir /s /b "c:\Program Files (x86)\Git\bin\git.exe');
    else
        search = 0;
        searchx86 = 0;
    end
    
    if (search||searchx86)
        % If git exists but the status is 0, then it means that it is
        % not in the path.
        result = 'git is not included in the path';
    else
        % If git is NOT installed, then this should end the function.
        result = sprintf('git is not installed\n%s\n',...
            'Download it at http://git-scm.com/download');
    end
else
    % Otherwise we can call the real git with the arguments
    arguments = parse(varargin{:});
    if ispc
        prog = '';
    else
        prog = ' | cat';
    end
    [err, result] = system([gitlocation, arguments, prog]);
end

% trim trailing newline character
if ~isempty(result) && strcmp(result(end), sprintf('\n'))
    result = result(1:end-1);
end

    % nested parse function
    function space_delimited_list = parse(varargin)
        space_delimited_list = cell2mat(...
            cellfun(@(s)([s,' ']),varargin,'UniformOutput',false));
    end

end %git sub-function
