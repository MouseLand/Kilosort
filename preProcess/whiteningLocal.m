function Wrot = whiteningLocal(CC, yc, xc, nRange)

Wrot = zeros(size(CC,1), size(CC,1));
for j = 1:size(CC,1)
    ds          = (xc - xc(j)).^2 + (yc - yc(j)).^2;
    [~, ilocal] = sort(ds, 'ascend');
    ilocal      = ilocal(1:nRange);
    
    [E, D]      = svd(CC(ilocal, ilocal));
    D           = diag(D);
    eps         = 1e-6;
    wrot0       = E * diag(1./(D + eps).^.5) * E';
    Wrot(ilocal, j)  = wrot0(:,1);
end