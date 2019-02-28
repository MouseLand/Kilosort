function [uS, idchan] = extractPCbatch2(rez, wPCA, ibatch, iC)

wPCA = gpuArray(single(wPCA(:, 1:3)));
% wPCA = gpuArray(single(rez.ops.wPCA(:, 1:3)));

ops = rez.ops;
% nPC = size(wPCA,2);

% indices of channels relative to center
NchanNear = size(iC,1);
% sigmaMask  = ops.sigmaMask;

Nbatch      = rez.temp.Nbatch;

NT  	= ops.NT;
batchstart = 0:NT:NT*Nbatch;

fid = fopen(ops.fproc, 'r');

offset = 2 * ops.Nchan*batchstart(ibatch);
fseek(fid, offset, 'bof');
dat = fread(fid, [NT ops.Nchan], '*int16');
fclose(fid);

% move data to GPU and scale it
if ops.GPU
    dataRAW = gpuArray(dat);
else
    dataRAW = dat;
end
dataRAW = single(dataRAW);
dataRAW = dataRAW / ops.scaleproc;

nt0min = rez.ops.nt0min;
spkTh = ops.ThPre;
[nt0, NrankPC] = size(wPCA);
[NT, Nchan] = size(dataRAW);

Params = [NT Nchan NchanNear nt0 nt0min spkTh NrankPC];

[uS, idchan] = mexThSpkPC(Params, dataRAW, wPCA, iC-1);

idchan = idchan + 1;
