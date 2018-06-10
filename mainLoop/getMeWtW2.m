function [WtW, iList] = getMeWtW2(W, U0, Nnearest)

[nt0, Nfilt, Nrank] = size(W);


WtW     = gpuArray.zeros(Nfilt,Nfilt, 'single');

for i = 1:Nrank
    for j = 1:Nrank
        utu0 = U0(:,:,i)' * U0(:,:,j);
        
        wtw0 = W(:,:,i)' * W(:,:,j);        
        
        WtW = WtW + wtw0.*utu0;
    end
end

if nargin>2 && nargout>1
    cc = max(WtW(:,:,:), [], 3);
    [~, isort] = sort(cc, 1, 'descend');
    
    iNear = rem([1:Nnearest]-1, Nfilt) + 1;
    iList = int32(gpuArray(isort(iNear, :)));
end