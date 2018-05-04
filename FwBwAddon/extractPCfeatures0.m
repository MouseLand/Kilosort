function [uprojDAT, call, muall, tall] = extractPCfeatures0(rez, wPCA)

ops = rez.ops;
nPC = size(wPCA,2);

call  = zeros(1, 5e6);
muall  = zeros(1, 5e6);
tall   = zeros(1, 5e6);

i0 = 0;

% indices of channels relative to center
nCH = 8;
dc = [-nCH:nCH];
% dc = -12:15;
%%
Nbatch      = rez.temp.Nbatch;


NT  	= ops.NT;
batchstart = 0:NT:NT*Nbatch;

uprojDAT = zeros(size(wPCA,2) * numel(dc),  1e7, 'single');

fid = fopen(ops.fproc, 'r');
for ibatch = 1:Nbatch   
    offset = 2 * ops.Nchan*batchstart(ibatch);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');
    
    if ibatch==1
        ioffset         = 0;
    else
        ioffset         = ops.ntbuff;
    end
    toff = -ioffset + (NT-ops.ntbuff)*(ibatch-1);
    
    % move data to GPU and scale it
    if ops.GPU
        dataRAW = gpuArray(dat);
    else
        dataRAW = dat;
    end
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;
    
    % find isolated spikes
    [row, col, mu] = isolated_peaks_new(dataRAW, ops);
    [~, isort] = sort(row);
    row = row(isort);
    col = col(isort);
    mu  = mu(isort);
    
    muall(1,i0 + (1:numel(col)))       = gather(mu);
    tall(1,i0 + (1:numel(row)))        = gather(row-ops.nt0) + toff;
    call(1,i0 + (1:numel(col)))        = gather(col);
    
    % get clips from upsampled data
    clips  = get_SpikeSample(dataRAW, row, col, dc);
    
    % compute center of mass of each spike and add to height estimate
    uS = reshape(wPCA' * clips(:, :), nPC , numel(dc), []);
%     uS = permute(uS, [2 1 3]);
    uS = reshape(uS, nPC*numel(dc), []);
    
    if i0+numel(row)>size(uprojDAT,2)
        nleft = size(uprojDAT,2) - i0;
        uprojDAT(:, i0 + (1:nleft)) = gather_try(uS(:, 1:nleft));
        i0 = i0 + nleft;
        break;
    end
    
    uprojDAT(:, i0 + (1:numel(row))) = gather_try(uS);    
    
    i0 = i0 + numel(row);
end

call(i0+1:end) = [];
tall(i0+1:end) = [];
muall(i0+1:end) = [];
uprojDAT(:, i0+1:end) = [];
fclose(fid);



