

function [ops, Wrot] = computeWhitening(ops)

ops.nt0 	= getOr(ops, {'nt0'}, 61);

[chanMap, xc, yc, kcoords, NchanTOTdefault] = loadChanMap(ops.chanMap);
ops.Nchan = numel(chanMap);
ops.NchanTOT = getOr(ops, 'NchanTOT', NchanTOTdefault);

NchanTOT = ops.NchanTOT;
NT       = ops.NT ;

rez.ops         = ops;
rez.xc = xc;
rez.yc = yc;

rez.xcoords = xc;
rez.ycoords = yc;

% rez.connected   = connected;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords; 

bytes = get_file_size(ops.fbinary);
nTimepoints = floor(bytes/NchanTOT/2);

rez.ops.tstart = ceil(ops.trange(1) * ops.fs); 
rez.ops.tend   = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); 

rez.ops.sampsToRead = rez.ops.tend-rez.ops.tstart; 

NTbuff      = NT + 4*ops.ntbuff;
% Nbatch      = ceil(rez.ops.sampsToRead /(NT-ops.ntbuff));
Nbatch = 2; % for the purposes of the gui, we only use two batches. It's not exactly the full whitening matrix, but it's close enough

% by how many bytes to offset all the batches
twind = rez.ops.tstart * NchanTOT*2;

%% load data into patches, filter, compute covariance
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
end
ops.b1 = b1; ops.a1 = a1;

fid = fopen(ops.fbinary, 'r');
Nchan = rez.ops.Nchan;
if ops.GPU
    CC = gpuArray.zeros( Nchan,  Nchan, 'single');
else
    CC = zeros( Nchan,  Nchan, 'single');
end

if ops.useRAM
    DATA = zeros(NT, rez.ops.Nchan, Nbatch, 'int16');
else
    DATA = [];
end

ibatch = 1;
while ibatch<=5  
    %drawnow; pause(0.05); 
    offset = max(0, twind + 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    fseek(fid, offset, 'bof');
    buff = fread(fid, [NchanTOT NTbuff], '*int16');
        
    if isempty(buff)
        break;
    end
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end
    
    datr = ksFilter(buff, ops);
    
    CC        = CC + (datr' * datr)/NT;    
    
    ibatch = ibatch + ops.nSkipCov;
end
CC = CC / ceil((Nbatch-1)/ops.nSkipCov);

fclose(fid);

if ops.whiteningRange<Inf
    %drawnow; pause(0.05); 
    ops.whiteningRange = min(ops.whiteningRange, Nchan);
    Wrot = whiteningLocal(gather_try(CC), yc, xc, ops.whiteningRange);
else
    [E, D] 	= svd(CC); drawnow; pause(0.05); 
    D       = diag(D); drawnow; pause(0.05); 
    eps 	= 1e-6; drawnow; pause(0.05); 
    Wrot 	= E * diag(1./(D + eps).^.5) * E';
end
Wrot    = ops.scaleproc * Wrot;
