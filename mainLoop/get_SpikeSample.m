function clips = get_SpikeSample(dataRAW, row, col, dc)

[nT, nChan] = size(dataRAW);

% times around the peak to consider
dt = -21 + [1:61];

% temporal indices
indsT = repmat(row', numel(dt), 1) + repmat(dt', 1, numel(row));
indsC = repmat(col', numel(dc), 1) + repmat(dc', 1, numel(col));

indsC(indsC<1)     = 1;
indsC(indsC>nChan) = nChan;

indsT = permute(indsT, [1 3 2]);
indsC = permute(indsC, [3 1 2]);
ix = indsT + (indsC-1) * nT;

% extract only spatial indices within the col index
clips = reshape(dataRAW(ix), numel(dt), numel(dc), numel(row));
