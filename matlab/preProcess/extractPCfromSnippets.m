function wPCA = extractPCfromSnippets(rez, nPCs)
% extracts principal components for 1D snippets of spikes from all channels
% loads a subset of batches to find these snippets

ops = rez.ops;
Nbatch      = rez.temp.Nbatch;
NT  	= ops.NT;
batchstart = 0:NT:NT*Nbatch;

% extract the PCA projections
CC = zeros(ops.nt0); % initialize the covariance of single-channel spike waveforms
fid = fopen(ops.fproc, 'r'); % open the preprocessed data file

for ibatch = 1:100:Nbatch % from every 100th batch
    offset = 2 * ops.Nchan*batchstart(ibatch);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');

    % move data to GPU and scale it back to unit variance
    dataRAW = gpuArray(dat);
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;

    % find isolated spikes from each batch
    [row, col, mu] = isolated_peaks_new(dataRAW, ops);

    % for each peak, get the voltage snippet from that channel
    clips = get_SpikeSample(dataRAW, row, col, ops, 0);

    c = sq(clips(:, :));
    CC = CC + gather(c * c')/1e3; % scale covariance down by 1,000 to maintain a good dynamic range
end
fclose(fid);

[U Sv V] = svdecon(CC); % the singular vectors of the covariance matrix are the PCs of the waveforms

wPCA = U(:, 1:nPCs); % take as many as needed

wPCA(:,1) = - wPCA(:,1) * sign(wPCA(21,1)); % adjust the arbitrary sign of the first PC so its negativity is downward

