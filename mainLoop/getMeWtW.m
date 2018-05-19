function [WtW, iList] = getMeWtW(W, U0, Nnearest)

[nt0, Nfilt, Nrank] = size(W);

Params = double([1 Nfilt 0 0 0 0 0 0 0 nt0]);

WtW     = gpuArray.zeros(Nfilt,Nfilt,2*nt0-1, 'single');
for i = 1:Nrank
    for j = 1:Nrank
        utu0 = U0(:,:,i)' * U0(:,:,j);
        wtw0 =  mexWtW2(Params, W(:,:,i), W(:,:,j), utu0);
        WtW = WtW + wtw0;
    end
end

if nargin>2 && nargout>1
    cc = WtW(:,:,nt0);
    [~, isort] = sort(cc, 1, 'descend');
    iList = int32(gpuArray(isort(1:Nnearest, :)));
end