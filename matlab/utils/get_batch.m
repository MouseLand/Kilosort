function dataRAW = get_batch(ops, ibatch)

Nbatch        = ops.Nbatch;
NT  	      = ops.NT;

batchstart = 0:NT:NT*Nbatch; % batches start at these timepoints
offset = 2 * ops.Nchan*batchstart(ibatch); % binary file offset in bytes

fid = fopen(ops.fproc, 'r');
fseek(fid, offset, 'bof');
dat = fread(fid, [NT ops.Nchan], '*int16');
fclose(fid);

% move data to GPU and scale it
dataRAW = gpuArray(dat);
dataRAW = single(dataRAW);
dataRAW = dataRAW / ops.scaleproc;

