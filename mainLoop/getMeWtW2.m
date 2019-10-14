function [WtW, iList] = getMeWtW2(W, U0, Nnearest)
% this function compute the correlation between any two pairs of templates
% it relies on the fact that the W and U0 are unit normalized, so that the product of a
% template with itself is 1, as it should be if we're trying to calculate correlations

% takes input the temporal and spatial factors of the low-rank template, as
% well as the number of most similar template pairs desired to be output in
% iList

[nt0, Nfilt, Nrank] = size(W);
WtW     = gpuArray.zeros(Nfilt,Nfilt, 'single');

% since the templates are factorized into orthonormal components, we can compute dot products
% one dimension at a time
for i = 1:Nrank
    for j = 1:Nrank
      %  this computes the spatial dot product
        utu0 = U0(:,:,i)' * U0(:,:,j);
      %  this computes the temporal dot product
        wtw0 = W(:,:,i)' * W(:,:,j);

      % the element-wise product of these is added to the matrix of correlatioons
        WtW = WtW + wtw0.*utu0;
    end
end

if nargin>2 && nargout>1
    % also return a list of most correlated template pairs
    [~, isort] = sort(WtW, 1, 'descend');

    iNear = rem([1:Nnearest]-1, Nfilt) + 1; % if we don't have enough templates yet, just wrap the indices around the range 1:Nfilt
    iList = int32(gpuArray(isort(iNear, :)));  % return the list of pairs for each template
end
