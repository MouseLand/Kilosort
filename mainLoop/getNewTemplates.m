function [Params, dWU, W, nsp, sd, derr, tstate] = ...
    getNewTemplates(ops,Params, dWU, W, data, wPCA, W0, nsp, sd, derr, tstate)

Nfilt = size(W,2);

% this adds templates
if (ops.Nfilt <= Nfilt)
    return;
end

Params(2) = Nfilt;
dWU0 = mexGetSpikes(Params, data, wPCA);

if size(dWU0,3)==0
    return;
end

dWU0 = reshape(wPCA * (wPCA' * dWU0(:,:)), size(dWU0));

Nnew = min(ops.Nfilt, Nfilt + size(dWU0,3)) - Nfilt;

dWU(:,:, Nfilt + [1:Nnew]) = dWU0(:,:,[1:Nnew]);

W(:,Nfilt + [1:Nnew],:) = W0(:,ones(1,Nnew),:);
nsp(Nfilt + [1:Nnew])   = ops.minFR * ops.NT/ops.fs;
sd(Nfilt +  [1:Nnew])   = 10^2;
derr(Nfilt+ [1:Nnew])   = 8^2;
% mu(Nfilt+ [1:Nnew])   = 10;
tstate(Nfilt+ [1:Nnew]) = 1;

Nfilt = size(W,2);
Params(2) = Nfilt;




