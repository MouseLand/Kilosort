function U = zeroOutKcoords(U, kcoords, criterionNoiseChannels)

[M, imax] = max(abs(U(:,:,1)), [], 1);

% determine over how many channel groups each template exists
aU = sum(U.^2,3).^.5;
ngroups = max(kcoords(:));

aUgroups = zeros(ngroups, size(U,2));
for j = 1:ngroups
    aUgroups(j, :) = mean(aU(kcoords==j,:), 1);
end

% the "effective" number of channel groups is defined below.
% for cases when X channel groups have equal non-zero weights, this number
% equals X
nEffective = sum(aUgroups,1).^2./sum(aUgroups.^2, 1);

[nEffSort, isort] = sort(nEffective, 'descend');

if criterionNoiseChannels<1
    % if this criterion is less than 1, it will be treated as a fraction 
    % of the total number of clusters
    nNoise = ceil(criterionNoiseChannels * size(U,2));
    ThLocal = nEffSort(nNoise);
else
    % if this criterion is larger than 1, it will be treated as the
    % effective number of channel groups at which to set the threshold
    ThLocal = criterionNoiseChannels;
end


for i = 1:size(U,2)
    if ThLocal > nEffective(i)
        U(kcoords~=kcoords(imax(i)),i,:) = 0;
        U(:,i,:) = normc(squeeze(U(:,i,:)));
    end
end

