function shift_batch_on_disk(rez, ibatch, shift, sig)
% this function finds threshold crossings in the data using
% projections onto the pre-determined principal components
% wPCA is number of time samples by number of PCs
% ibatch is a scalar indicating which batch to analyze
% iC is NchanNear by Nchan, indicating for each channel the nearest
% channels to it


% ibatch = 1;
% shift = dshift(ibatch);

ops = rez.ops;
Nbatch      = rez.temp.Nbatch;
NT  	      = ops.NT;


batchstart = 0:NT:NT*Nbatch; % batches start at these timepoints
offset = 2 * ops.Nchan*batchstart(ibatch); % binary file offset in bytes


fclose all;
fid = fopen(ops.fproc, 'r+');
fseek(fid, offset, 'bof');
dat = fread(fid, [NT ops.Nchan], '*int16');

xp = cat(2, rez.xc, rez.yc);
% sig = 15;

Kxx = kernel2D(xp, xp, sig);
yp = xp;
yp(:, 2) = yp(:, 2) - shift; % * sig;

Kyx = kernel2D(yp, xp, sig);

M = Kyx /(Kxx + .01 * eye(size(Kxx,1)));
% M = (rez.Wrot * M) / rez.Wrot;

dati = gpuArray(single(dat)) * gpuArray(M)';

dat_cpu = gather(int16(dati));
fseek(fid, offset, 'bof');
fwrite(fid, dat_cpu, 'int16'); % write this batch to binary file

fclose(fid);

if ~isempty(getOr(ops, 'fbinaryproc', []))
    fid2 = fopen(ops.fbinaryproc, 'a');
    ifirst = ops.ntbuff+1;
    ilast = ops.NT;
    if ibatch==1
        ifirst = 1;
        ilast = ops.NT-ops.ntbuff;
    end
    fwrite(fid2, dat_cpu(ifirst:ilast, :)', 'int16');
    fclose(fid2);
end