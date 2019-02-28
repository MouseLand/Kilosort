wPCA = gpuArray(single(rez.ops.wPCA(:, 1:3)));

ops = rez.ops;
nPC = size(wPCA,2);

% indices of channels relative to center
NchanNear = 2*8+1;
sigmaMask  = ops.sigmaMask;

[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear);
%%
ibatch = 1;
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
%%
nt0min = rez.ops.nt0min;
spkTh = ops.ThPre;
[nt0, NrankPC] = size(wPCA);
[NT, Nchan] = size(dataRAW);

Params = [NT Nchan NchanNear nt0 nt0min spkTh NrankPC];

[uS, idchan] = mexThSpkPC(Params, dataRAW, wPCA, iC-1);

%%
dProc = mexFilterPCs(Params, dataRAW, wPCA(:, 1:3));
dProc = sqrt(max(0, dProc));

[row, col, mu] = isolated_peaks_PC(dProc, ops);

[~, isort] = sort(row);
row = row(isort);
col = col(isort);
mu  = mu(isort);

% get clips from upsampled data
clips  = get_SpikeSample(dataRAW, row, col, ops, dc, 1);

% compute center of mass of each spike and add to height estimate
uS = reshape(wPCA' * clips(:, :), nPC , numel(dc), []);
%     uS = permute(uS, [2 1 3]);
uS = reshape(uS, nPC*numel(dc), []);

fclose(fid);


%%

[uS, idchan] = extractPCbatch(rez, rez.ops.wPCA, 1, iC);