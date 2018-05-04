function  [W, U, mu, UtU] = update_params(mu, W, U, dWUtot, nspikes)

[Nchan, Nfilt, Nrank] = size(U);

dWUtotCPU = gather_try(dWUtot);
ntot = sum(nspikes,2);

for k = 1:Nfilt
    if ntot(k)>5
        
        [Uall, Sv, Vall] = svd(gather_try(dWUtotCPU(:,:,k)), 0);
        Sv = diag(Sv);
        sumSv2 = sum(Sv(1:Nrank).^2).^.5;
        for irank = 1:Nrank
            [~, imax] = max(abs(Uall(:,irank)), [], 1);
            W(:,k,irank) = - Uall(:,irank) * sign(Uall(imax,irank)) * Sv(irank)/sumSv2;
            U(:,k,irank) = - Vall(:,irank) * sign(Uall(imax,irank));
        end
        mmax = max(abs(U(:,k,1)));
        Usize = squeeze(abs(U(:,k,:)));
        Usize = Usize .* repmat(Sv(1:Nrank)'/Sv(1), Nchan, 1);
        ibad = max(Usize, [], 2) < .1 * mmax;
        
        U(ibad,k,:) = 0;
    end
end

% mu = zeros(Nfilt,1, 'single');
for k = 1:Nfilt
    if ntot(k)>5
        wu = squeeze(W(:,k,:)) * squeeze(U(:,k,:))';
        mu(k) = sum(sum(wu.*squeeze(dWUtotCPU(:,:,k))));
    end
end

for k = 1:Nfilt
    if ntot(k)>5
        wu = squeeze(W(:,k,:)) * squeeze(U(:,k,:))';
        newnorm = sum(wu(:).^2).^.5;
        W(:,k,:) = W(:,k,:)/newnorm;
    end
end

% compute adjacency matrix UtU
U(isnan(U)) = 0;
U0 = gpuArray(U);
utu = gpuArray.zeros(Nfilt, 'single');
for irank = 1:Nrank
    utu = utu + (U0(:,:,irank)' * U0(:,:,irank));
end

UtU = logical(utu);
