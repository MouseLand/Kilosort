function Us = get_PCproj(S1, row, col, wPCA, maskMaxChans)

[nT, nChan] = size(S1);
dt = -21 + [1:size(wPCA,1)];
inds = repmat(row', numel(dt), 1) + repmat(dt', 1, numel(row));

clips = reshape(S1(inds, :), numel(dt), numel(row), nChan);


mask = repmat([1:nChan], [numel(row) 1]) - repmat(col, 1, nChan);
Mask(1,:,:) = abs(mask)<maskMaxChans;

clips = bsxfun(@times, clips , Mask);

Us = wPCA' * reshape(clips, numel(dt), []);
Us = reshape(Us, size(wPCA,2), numel(row), nChan);

Us = permute(Us, [3 2 1]);