function wPCA = extractPCfromSnippets(rez, nPCs)

ops = rez.ops;

% Nchan 	= ops.Nchan;

Nbatch      = rez.temp.Nbatch;

NT  	= ops.NT;
batchstart = 0:NT:NT*Nbatch;

% extract the PCA projections
CC = zeros(61);
fid = fopen(ops.fproc, 'r');

for ibatch = 1:100:Nbatch
    offset = 2 * ops.Nchan*batchstart(ibatch);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');
    
    % move data to GPU and scale it
    if ops.GPU
        dataRAW = gpuArray(dat);
    else
        dataRAW = dat;
    end
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;
    
    
    % find isolated spikes
    [row, col, mu] = isolated_peaks_new(dataRAW, ops);
    
    clips = get_SpikeSample(dataRAW, row, col, 0);
    
    c = sq(clips(:, :));
    CC = CC + gather(c * c')/1e3;
    
end
fclose(fid);

[U Sv V] = svdecon(CC);

wPCA = U(:, 1:nPCs);
