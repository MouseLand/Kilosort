function [U,S,V] = svdecon(X, nPC0)
% Input:
% X : m x n matrix
%
% Output:
% X = U*S*V'
%
% Description:
% Does equivalent to svd(X,'econ') but faster
%
% Vipin Vijayan (2014)

%X = bsxfun(@minus,X,mean(X,2));
[m,n] = size(X);

nPC = min(m,n);
if nargin>1
   nPC = nPC0; 
end

if  m <= n
    C = X*X';
    [U,D] = eig(C);
    
    clear C;
    
    [d,ix] = sort(abs(diag(D)),'descend');
    U = U(:,ix);    
    d = gather(d(1:nPC));
    
    U = gather(U(:, 1:nPC));
    
    if nargout > 2
        V = X'*U;
        s = sqrt(d);
        V = bsxfun(@(x,c)x./c, V, s');
        S = diag(s);
        
        V = gather(V);
    end
else
    C = X'*X; 
    [V,D] = eig(C);
    D = gather(D);
    V = gather(V);
    clear C;
    
    [d,ix] = sort(abs(diag(D)),'descend');
    V = V(:,ix);    
    
    U = X*V; % convert evecs from X'*X to X*X'. the evals are the same.
    %s = sqrt(sum(U.^2,1))';
    s = sqrt(d);
    U = bsxfun(@(x,c)x./c, U, s');
    S = diag(s);
end
