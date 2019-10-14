function [WtW, iList] = getMeWtW(W, U0, Nnearest)
% this function computes correlation between templates at ALL timelags from each other
% takes the max over timelags to obtain a similarity score
% also returns lists of most similar templates to each template
% takes as input the low-rank factorization of templates (W for time and U0
% for space)

[nt0, Nfilt, Nrank] = size(W); % W is timesamples (default = 61 ), by number of templates, by rank (default = 3)

Params = double([1 Nfilt 0 0 0 0 0 0 0 nt0]);

% initialize correlation matrix for all timelags
WtW     = zeros(Nfilt,Nfilt,2*nt0-1, 'single');
for i = 1:Nrank
    for j = 1:Nrank
        % the dot product factorizes into separable products for each spatio-temporal component
        utu0 = U0(:,:,i)' * U0(:,:,j); % spatial products
        wtw0 =  mexWtW2(Params, W(:,:,i), W(:,:,j), utu0); % temporal convolutions get multiplied wit hthe spatial products
        wtw0 = gather(wtw0); % take this matrix off the GPU (it's big)
        WtW = WtW + wtw0; % add it to the full correlation array
    end
end

if nargin>2 && nargout>1
    % the maximum across timelags accounts for sample alignment mismatch
    cc = max(WtW(:,:,:), [], 3);

    [~, isort] = sort(cc, 1, 'descend');
    iNear = rem([1:Nnearest]-1, Nfilt) + 1;% if we don't have enough templates yet, just wrap the indices around the range 1:Nfilt
    iList = int32(gpuArray(isort(iNear, :))); % return the list of pairs for each template
end
