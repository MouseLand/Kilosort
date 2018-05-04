function Minv = my_inv(M, eps)

[U, Sv, V] = svd(M);

Sv = max(Sv, eps); 

Minv = U * diag(1./diag(Sv)) * V';