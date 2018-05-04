
S = sparse(ceil(rez.st/3e4), rez.clu, ones(1, numel(rez.clu)));
S(1, rez.ops.Nfilt) = 0;

S = gpuArray(single(full(S)));

% Shigh = S - mean(S,1);
% Shigh = S - my_conv2(S,500,1);

Slow = my_conv2(S,500,1);

rat = min(Slow, [], 1) ./max(Slow, [],1);

S0 = S(:, rat>.5);

% [U Sv V] = svdecon(zscore(Shigh, 1, 1));
% [U Sv V] = svdecon(S0 - mean(S0, 1));
[U Sv V] = svdecon(zscore(S0,1,1));
%%
clf
plot(U(:,4))
%%
imagesc(S0, [0 20])

%%

W = reshape(rez.W, 7, 374, []);

Wrec = reshape(rez.wPCA * W(:,:), 61, 374, []);


imagesc(Wrec(:,:, 244)')

%%
1