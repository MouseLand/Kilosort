function Wrot = whiteningFromCovariance(CC)

[E, D] 	= svd(CC); % covariance eigendecomposition (same as svd for positive-definite matrix)
D       = diag(D); % take the non-zero values from the diagonal
eps 	= 1e-6;
Wrot 	= E * diag(1./(D + eps).^.5) * E'; % this is the symmetric whitening matrix (ZCA transform)
