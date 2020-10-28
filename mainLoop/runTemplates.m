function rez = runTemplates(rez)

% this function will run Kilosort2 initializing at some previously found
% templates. These must be specified in rez.W, rez.U and rez.mu. The batch
% number at which the new template run is started is by default 1 (modify it by changing rez.istart). 

% If you don't mind changes in template throughout the recording, but want
% to use the results from a previous session after splitting and merging, just pass
% the rez output from this previous session.

% Keep in mind that these templates will nonetheless
% change substantially as they track neurons throughout the new recording. 
% As such, it is best to use the templates obtained at the very end of the
% previous recording, which are found in rez.WA(:,:,:,end),
% rez.UA(:,:,:,end), rez.muA(:,end). Keep in mind that any merging and
% splitting steps are done after running the templates, and would result in
% difference between sessions if done separately on each session. 

% update: these recommendations no longer apply for the datashift version!

if sum(isfield(rez, {'istart'}))<1
    warning('if using pre-loaded templates, please specify istart, defaulting to 1');
    rez.istart = 1;
end

if sum(isfield(rez, {'W', 'U', 'mu'}))<3
    error('missing at least one field: W, U, mu');
end

Nbatches = rez.ops.Nbatch;

iorder = 1:Nbatches;

[rez, st3, fW,fWpc] = trackAndSort(rez, iorder);

% sort all spikes by batch -- to keep similar batches together,
% which avoids false splits in splitAllClusters. Break ties 
% [~, isort] = sortrows(st3,[5,1,2,3,4]); 
% st3 = st3(isort, :);
% fW = fW(:, isort);
% fWpc = fWpc(:, :, isort);

% just display the total number of spikes
fprintf( 'Number of spikes before applying cutoff: %d\n', size(st3,1));

rez.st3 = st3;
rez.st2 = st3; % keep also an st2 copy, because st3 will be over-written by one of the post-processing steps

% the template features are stored in cProj, like in Kilosort1
rez.cProj    = fW';

%  permute the PC projections in the right order
rez.cProjPC     = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');
% iNeighPC keeps the indices of the channels corresponding to the PC features

