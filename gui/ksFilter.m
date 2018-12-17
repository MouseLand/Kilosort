
function datr = ksFilter(buff, ops)

if ~isfield(ops, 'b1')
    if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
        [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
    else
        [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
    end
else
    b1 = ops.b1; a1 = ops.a1;
end

if ops.GPU
    dataRAW = gpuArray(buff);
else
    dataRAW = buff;
end
dataRAW = dataRAW';
dataRAW = single(dataRAW);
dataRAW = dataRAW(:, ops.chanMap.chanMap);

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