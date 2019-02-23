function Wrot = whiteningLocal(CC, yc, xc, nRange)

Wrot = zeros(size(CC,1), size(CC,1));
for j = 1:size(CC,1)
    ds          = (xc - xc(j)).^2 + (yc - yc(j)).^2;
    [~, ilocal] = sort(ds, 'ascend');
    ilocal      = ilocal(1:nRange);
    
    [E, D]      = svd(CC(ilocal, ilocal));
    D           = diag(D);
%     eps 	= mean(D); %1e-6;
    eps 	= 1e-6;
    f  = mean((D+eps) ./ (D+1e-6)).^.5;
%     fprintf('%2.2f ', f)
    wrot0       = E * diag(f./(D + eps).^.5) * E';
    Wrot(ilocal, j)  = wrot0(:,1);
end
