tic
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
for iNN = 1:size(rez.W,2)
     stimes{iNN} = rez.st3(rez.st3(:,2)==iNN,1);
end
%stimes = gtimes;

Wraw = zeros(nt0, Nchans, numel(stimes));
for ibatch = 1:Nbatch    
    if ibatch>Nbatch_buff
        offset = 2 * ops.Nchan*batchstart(ibatch-Nbatch_buff); % - ioffset;
        fseek(fid, offset, 'bof');
        dat = fread(fid, [NT ops.Nchan], '*int16');
    else
       dat = DATA(:,:,ibatch); 
    end
    dataRAW = gpuArray(dat);
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;
        
    
    if ibatch==1; ioffset = 0;
    else ioffset = ops.ntbuff;
    end
    
    for iNN = 1:numel(stimes)
        st = stimes{iNN} + ioffset - (NT-ops.ntbuff)*(ibatch-1) - 20;
        st(st<0) = [];
        st(st>NT-ops.ntbuff) = [];
        
        if ~isempty(st)
            inds = repmat(st', nt0, 1) + repmat(ts, 1, numel(st));
            
            Wraw(:,:,iNN) = Wraw(:,:,iNN) + ...
                gather_try(squeeze(sum(reshape(dataRAW(inds, :), nt0, numel(st), Nchans),2)));
        end
    end
    
end

for iNN = 1:numel(stimes)
     Wraw(:,:,iNN) = Wraw(:,:,iNN)/numel(stimes{iNN});
end
fprintf('Time %3.2f. Mean waveforms computed... \n', toc);






