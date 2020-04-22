function shift_batch_on_disk(rez, ibatch, shift)
% this function finds threshold crossings in the data using
% projections onto the pre-determined principal components
% wPCA is number of time samples by number of PCs
% ibatch is a scalar indicating which batch to analyze
% iC is NchanNear by Nchan, indicating for each channel the nearest
% channels to it

ops = rez.ops;
Nbatch      = rez.temp.Nbatch;
NT  	      = ops.NT;


batchstart = 0:NT:NT*Nbatch; % batches start at these timepoints
offset = 2 * ops.Nchan*batchstart(ibatch); % binary file offset in bytes

fid = fopen(ops.fproc, 'r+');
fseek(fid, offset, 'bof');
dat = fread(fid, [NT ops.Nchan], '*int16');

iChan = 1:ops.Nchan;
iChanS = mod(iChan + 2 * shift - 1, ops.Nchan) + 1;
dat(:, iChanS) = dat;

fseek(fid, offset, 'bof');
fwrite(fid, dat, 'int16'); % write this batch to binary file

fclose(fid);
