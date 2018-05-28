function [uS, col, mu, row] = extractPCbatch(rez, wPCA, ibatch)

ops = rez.ops;
nPC = size(wPCA,2);

% indices of channels relative to center
nCH = 8;
dc = [-nCH:nCH];
% dc = -12:15;
%%
Nbatch      = rez.temp.Nbatch;

NT  	= ops.NT;
batchstart = 0:NT:NT*Nbatch;

fid = fopen(ops.fproc, 'r');

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
wPCA = gpuArray(wPCA);

[nt0 Nrank] = size(wPCA);
[NT Nchan] = size(dataRAW);

Params = [NT Nchan nt0 3];

dProc = mexFilterPCs(Params, dataRAW, wPCA(:, 1:3));
dProc = sqrt(max(0, dProc));

[row, col, mu] = isolated_peaks_PC(dProc, ops);

[~, isort] = sort(row);
row = row(isort);
col = col(isort);
mu  = mu(isort);

% get clips from upsampled data
clips  = get_SpikeSample(dataRAW, row, col, dc);

% compute center of mass of each spike and add to height estimate
uS = reshape(wPCA' * clips(:, :), nPC , numel(dc), []);
%     uS = permute(uS, [2 1 3]);
uS = reshape(uS, nPC*numel(dc), []);




