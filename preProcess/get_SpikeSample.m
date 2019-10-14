function clips = get_SpikeSample(dataRAW, row, col, ops, dc, flag)
% given a batch of data (time by channels), and some time (row) and channel (col) indices for spikes, 
% this function returns the 1D time clips of voltage around those spike times

[nT, nChan] = size(dataRAW);

% times around the peak to consider
dt = [1:ops.nt0];

if nargin<6 || flag == 0
    dt = -ops.nt0min + dt; % the negativity is expected at nt0min, so we align the detected peaks there
end

% temporal indices (awkward way to index into full matrix of data)
indsT = row' + dt'; % broadcasting
indsC = col' + dc';

indsC(indsC<1)     = 1; % anything that's out of bounds just gets set to the limit
indsC(indsC>nChan) = nChan; % only needed for channels not time (due to time buffer)

indsT = permute(indsT, [1 3 2]);
indsC = permute(indsC, [3 1 2]);
ix = indsT + (indsC-1) * nT; % believe it or not, these indices grab just the right timesamples for our spikes

% grab the data and reshape it appropriately (time samples  by channels by num spikes)
clips = reshape(dataRAW(ix), numel(dt), numel(dc), numel(row));
