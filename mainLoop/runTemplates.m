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



if sum(isfield(rez, {'istart'}))<1
    warning('if using pre-loaded templates, please specify istart, defaulting to 1');
    rez.istart = 1;
end

if sum(isfield(rez, {'W', 'U', 'mu'}))<3
    error('missing at least one field: W, U, mu');
end

istart = rez.istart;
Nbatches = numel(rez.iorig);

ihalf = find(rez.iorig==istart);

iorder_sorted = ihalf:-1:1;
iorder = rez.iorig(iorder_sorted);  %batch number in full set

[rez, st3_0, fW_0,fWpc_0] = trackAndSort(rez, iorder);

st3_0(:,5) = iorder_sorted(st3_0(:,5));

iorder_sorted = (ihalf+1):Nbatches;
iorder = rez.iorig(iorder_sorted);

[rez, st3_1, fW_1,fWpc_1] = trackAndSort(rez, iorder);

st3_1(:,5) = iorder_sorted(st3_1(:,5)); %batch number in full set

st3     = cat(1, st3_0, st3_1);
fW      = cat(2, fW_0, fW_1);
fWpc    = cat(3, fWpc_0, fWpc_1);

% sort all spikes by batch -- to keep similar batches together,
% which avoids false splits in splitAllClusters. Break ties 
[~, isort] = sortrows(st3,[5,1,2,3,4]); 
st3 = st3(isort, :);
fW = fW(:, isort);
fWpc = fWpc(:, :, isort);

% just display the total number of spikes
fprintf( 'Number of spikes before applying cutoff: %d\n', size(st3,1));

rez.st3 = st3;
rez.st2 = st3; % keep also an st2 copy, because st3 will be over-written by one of the post-processing steps

% the template features are stored in cProj, like in Kilosort1
rez.cProj    = fW';

%  permute the PC projections in the right order
rez.cProjPC     = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');
% iNeighPC keeps the indices of the channels corresponding to the PC features


% this whole next block is just done to compress the compressed templates
% we separately svd the time components of each template, and the spatial components
% this also requires a careful decompression function, available somewhere in the GUI code
Nrank   = 3; % this one is the rank of the templates
Nfilt = size(rez.W,2);
Nchan = rez.ops.Nchan;
nt0 = rez.ops.nt0;

nKeep = min([Nchan*3,Nbatches,20]); % how many PCs to keep
rez.W_a = zeros(nt0 * Nrank, nKeep, Nfilt, 'single');
rez.W_b = zeros(Nbatches, nKeep, Nfilt, 'single');
rez.U_a = zeros(Nchan* Nrank, nKeep, Nfilt, 'single');
rez.U_b = zeros(Nbatches, nKeep, Nfilt, 'single');
for j = 1:Nfilt
    % do this for every template separately
    WA = reshape(rez.WA(:, j, :, :), [], Nbatches);
    WA = gpuArray(WA); % svd on the GPU was faster for this, but the Python randomized CPU version might be faster still
    [A, B, C] = svdecon(WA);
    % W_a times W_b results in a reconstruction of the time components
    rez.W_a(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.W_b(:,:,j) = gather(C(:, 1:nKeep));
    
    UA = reshape(rez.UA(:, j, :, :), [], Nbatches);
    UA = gpuArray(UA);
    [A, B, C] = svdecon(UA);
    % U_a times U_b results in a reconstruction of the time components
    rez.U_a(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.U_b(:,:,j) = gather(C(:, 1:nKeep));
end

fprintf('Finished compressing time-varying templates \n')