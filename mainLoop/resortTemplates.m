function  [W, dWU, nsp, sd, derr, iW] = ...
    resortTemplates(nt0min, W, dWU, nsp, sd, derr, tstate)

% tstate says that some templats don't move! 

ix = find(tstate>.5);
% Nfilt = size(W,2);
% ix = 1:Nfilt;

[~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
iW = int32(squeeze(iW));

[~, isort] = sort(iW(ix));

iy = ix(isort);

iW(ix) = iW(iy);

W(:,ix,:) = W(:,iy, :);
dWU(:,:,ix) = dWU(:,:,iy);
nsp(ix) = nsp(iy);
sd(ix)  = sd(iy);
derr(ix) = derr(iy);