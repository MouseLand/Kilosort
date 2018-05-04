tic
nt0 = ops.nt0;

if ~isempty(ops.chanMap)
    load(ops.chanMap);
    chanMapConn = chanMap(connected>1e-6);
else
    chanMapConn = 1:ops.Nchan;
end
batch_path = fullfile(root, 'batches');
if ~exist(batch_path, 'dir')
    mkdir(batch_path);
end
NchanTOT = ops.NchanTOT;

d = dir(fullfile(root, fname));
ops.sampsToRead = floor(d.bytes/NchanTOT/2);


NT          = 128*1024+ ops.ntbuff;
NTbuff      = NT + 4*ops.ntbuff;
Nbatch      = ceil(d.bytes/2/NchanTOT /(NT-ops.ntbuff));

% load data into patches, filter, compute covariance, write back to
% disk

fprintf('Time %3.0fs. Loading raw data... \n', toc);
fid = fopen(fullfile(root, fname), 'r');
ibatch = 0;
Nchan = ops.Nchan;

Nchans = ops.Nchan;
ts = [1:1:nt0]';

clear stimes
% for iNN = 1:size(rez.W,2)
%     stimes{iNN} = rez.st3pos(rez.st3pos(:,2)==iNN,1);
% end
stimes = gtimes;

Wraw = zeros(nt0, Nchans, numel(stimes));

while 1
    ibatch = ibatch + 1;
    
    offset = max(0, 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    if ibatch==1
        ioffset = 0;
    else
        ioffset = ops.ntbuff;
    end
    fseek(fid, offset, 'bof');
    buff = fread(fid, [NchanTOT NTbuff], '*int16');
    
    if isempty(buff)
        break;
    end
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end
    
    offset = (NT-ops.ntbuff)*(ibatch-1)-64 - 40;
    
    buff = buff(chanMapConn,:)';
    %
    for iNN = 1:numel(stimes)
        st = stimes{iNN} - offset;
        st(st<0) = [];
        st(st>NT-ops.ntbuff) = [];
        
        if ~isempty(st)
            inds = repmat(st', nt0, 1) + repmat(ts, 1, numel(st));
            
            Wraw(:,:,iNN) = Wraw(:,:,iNN) + ...
                squeeze(sum(reshape(buff(inds, :), nt0, numel(st), Nchans),2));
        end
    end
    
end
fclose(fid);

for iNN = 1:numel(stimes)
     Wraw(:,:,iNN) = Wraw(:,:,iNN)/numel(stimes{iNN});
end
fprintf('Time %3.2f. Mean waveforms computed... \n', toc);






