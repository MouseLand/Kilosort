function [ccb1, isort, xs] = sortBatches2(ccb0)
% takes as input a matrix of nBatches by nBatches containing
% dissimilarities.
% outputs a matrix of sorted batches, and the sorting order, such that
% ccb1 = ccb0(isort, isort)

% put this matrix on the GPU
ccb0 = gpuArray(ccb0);

% compute its svd on the GPU (this might also be fast enough on CPU)
[u, s, v] = svdecon(ccb0);

% initialize the positions xs of the batch embeddings to be very small but proportional to the first PC
xs = .01 * u(:,1)/std(u(:,1));

% 200 iterations of gradient descent should be enough
niB = 200;

% this learning rate should usually work fine, since it scales with the average gradient
% and ccb0 is z-scored
eta = 1;
for k = 1:niB
    % euclidian distances between 1D embedding positions
    ds = (xs - xs').^2;
    % the transformed distances go through this function
    W  = log(1 + ds);

    % the error is the difference between ccb0 and W
    err = ccb0 - W;

    % ignore the mean value of ccb0
    err = err - mean(err(:));

    % backpropagate the gradients
    err = err./(1+ds);
    err2 = err .* (xs - xs');
    D = mean(err2, 2); % one half of the gradients is along this direction
    E = mean(err2, 1); % the other half is along this direction
    % we don't need to worry about the gradients for the diagonal because those are 0

    % final gradients for the embedding variable
    dx = -D + E';

    % take a gradient step
    xs = xs - eta * dx;
end

% sort the embedding positions xs
[~, isort] = sort(xs);

% sort the matrix of dissimilarities
ccb1 = ccb0(isort, isort);
