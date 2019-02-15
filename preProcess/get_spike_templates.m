% function [wTEMP, wPCA] = get_spike_templates(rez, nPCs)

ops = rez.ops;
Nbatch = ops.Nbatch;
twind = ops.twind;
NchanTOT = ops.NchanTOT;
NT = ops.NT;
NTbuff = ops.NTbuff;
chanMap = ops.chanMap;
Nchan = rez.ops.Nchan;
xc = rez.xc;
yc = rez.yc;

% load data into patches, filter, compute covariance
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
end

fid = fopen(ops.fbinary, 'r');

% irange = [NT/8:(NT-NT/8)];

ibatch = 1;
dd = gpuArray.zeros(61, 5e4, 'single');
ich = gpuArray.zeros(5e4,1, 'int16');
k = 0;

while ibatch<=Nbatch    
    offset = twind + 2*NchanTOT*NT* (ibatch-1);
    fseek(fid, offset, 'bof');
    buff = fread(fid, [NchanTOT NTbuff], '*int16');
        
    if isempty(buff)
        break;
    end
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end
    if ops.GPU
        dataRAW = gpuArray(buff);
    else
        dataRAW = buff;
    end
    dataRAW = dataRAW';
    dataRAW = single(dataRAW);
    dataRAW = dataRAW(:, chanMap);
    
    % subtract the mean from each channel
    dataRAW = dataRAW - mean(dataRAW, 1);
    
    datr = filter(b1, a1, dataRAW);
    datr = flipud(datr);
    datr = filter(b1, a1, datr);
    datr = flipud(datr);

    % CAR, common average referencing by median
    if getOr(ops, 'CAR', 1)
        datr = datr - median(datr, 2);
    end
    
    
    
    % determine any threshold crossings
    datr = datr./std(datr,1,1);
    
    mdat = my_min(datr, 30, 1);
    ind = find(datr<mdat+1e-3 & datr<ops.spkTh);
    [xi, xj] = ind2sub(size(dat), ind);
    ibad = xi<nt0 | xi>NT-nt0;
    xi(ibad) = [];
    xj(ibad) = [];
    
    clips = get_SpikeSample(datr, xi, xj, 0);    
    
    c = sq(clips(:, :));        
    if k+size(c,2)>size(dd,2)
        ich(2*size(dd,2)) = 0;
        dd(:, 2*size(dd,2)) = 0;
    end    
    dd(:, k + [1:size(c,2)]) = c;    
    ich(k + [1:size(c,2)]) = xj;
    
    k = k + size(c,2);
    
    ibatch = ibatch + ops.nSkipCov;
end

dd = dd(:, 1:k);
ich = ich(1:k);
fprintf('found %d threshold crossings \n', k)
%%
wTEMP = dd(:, randperm(size(dd,2), nPCs));
wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5;

for i = 1:10
   cc = wTEMP' * dd;
   [amax, imax] = max(cc,[],1);
   for j = 1:nPCs
      wTEMP(:,j)  = dd(:,imax==j) * amax(imax==j)';
   end
   wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5;
end

CC = dd * dd';
[U Sv V] = svdecon(CC);
wPCA = U(:, 1:nPCs);
wPCA(:,1) = - wPCA(:,1) * sign(wPCA(21,1));


fclose(fid);

