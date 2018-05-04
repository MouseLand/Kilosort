function  [W, U, mu, UtU, nu] = decompose_dWU(ops, dWU, Nrank, kcoords)

[nt0 Nchan Nfilt] = size(dWU);

W = zeros(nt0, Nrank, Nfilt, 'single');
U = zeros(Nchan, Nrank, Nfilt, 'single');
mu = zeros(Nfilt, 1, 'single');
% dmax = zeros(Nfilt, 1);

dWU(isnan(dWU)) = 0;
if ops.parfor
    parfor k = 1:Nfilt
        [W(:,:,k), U(:,:,k), mu(k)] = get_svds(dWU(:,:,k), Nrank);
    end
else
    for k = 1:Nfilt
        [W(:,:,k), U(:,:,k), mu(k)] = get_svds(dWU(:,:,k), Nrank);
    end
end
U = permute(U, [1 3 2]);
W = permute(W, [1 3 2]);

U(isnan(U)) = 0;

if numel(unique(kcoords))>1
    U = zeroOutKcoords(U, kcoords, ops.criterionNoiseChannels);
end

UtU = abs(U(:,:,1)' * U(:,:,1)) > .1;


Wdiff = cat(1, W, zeros(2, Nfilt, Nrank)) - cat(1,  zeros(2, Nfilt, Nrank), W);
nu = sum(sum(Wdiff.^2,1),3);
nu = nu(:);



% mu = min(mu, 200);