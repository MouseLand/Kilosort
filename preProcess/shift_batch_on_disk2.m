function [dat_cpu, dat, shifts] = shift_batch_on_disk2(rez, ibatch, shifts, ysamp, sig)
% register one batch of a whitened binary file

ops = rez.ops;
Nbatch      = rez.temp.Nbatch;
NT  	      = ops.NT;

batchstart = 0:NT:NT*Nbatch; % batches start at these timepoints
offset = 2 * ops.Nchan*batchstart(ibatch); % binary file offset in bytes

% upsample the shift for each channel using interpolation
if length(ysamp)>1
    shifts = interp1(ysamp, shifts, rez.yc, 'makima', 'extrap');
end
% load the batch
fclose all;
fid = fopen(ops.fproc, 'r+');
fseek(fid, offset, 'bof');
dat = fread(fid, [NT ops.Nchan], '*int16');

% 2D coordinates for interpolation 
xp = cat(2, rez.xc, rez.yc);

% 2D kernel of the original channel positions 
Kxx = kernel2D(xp, xp, sig);
% 2D kernel of the new channel positions
yp = xp;
yp(:, 2) = yp(:, 2) - shifts; % * sig;
Kyx = kernel2D(yp, xp, sig);

% kernel prediction matrix
M = Kyx /(Kxx + .01 * eye(size(Kxx,1)));

% the multiplication has to be done on the GPU
dati = gpuArray(single(dat)) * gpuArray(M)';

dat_cpu = gather(int16(dati));

if ~isempty(getOr(ops, 'fbinaryproc', []))
    % if the user wants to have a registered version of the binary file 
    % this one is not split into batches
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

if nargout==0
    % normally we want to write the aligned data back to the same file
    fseek(fid, offset, 'bof');
    fwrite(fid, dat_cpu, 'int16'); % write this batch to binary file
end
fclose(fid);
    
