function datr = gpufilter(buff, ops)

dataRAW = gpuArray(buff); % move int16 data to GPU
dataRAW = dataRAW';
dataRAW = single(dataRAW); % convert to float32 so GPU operations are fast
dataRAW = dataRAW(:, chanMap); % subsample only good channels

% subtract the mean from each channel
dataRAW = dataRAW - mean(dataRAW, 1); % subtract mean of each channel

% CAR, common average referencing by median
if getOr(ops, 'CAR', 1)
    dataRAW = dataRAW - median(dataRAW, 2); % subtract median across channels
end

% next four lines should be equivalent to filtfilt (which cannot be used because it requires float64)
datr = filter(b1, a1, dataRAW); % causal forward filter
datr = flipud(datr); % reverse time
datr = filter(b1, a1, datr); % causal forward filter again
datr = flipud(datr); % reverse time back
