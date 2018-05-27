function [W, U, dWU, mu, nsp] = triageTemplates(ops, W, U, dWU, mu, nsp, flag_merge)

m0 = ops.minFR * ops.NT/ops.fs;

idrop = nsp<m0;

W(:,idrop,:) = [];
U(:,idrop,:) = [];
dWU(:,:, idrop) = [];
mu(idrop) = [];
nsp(idrop) = [];
% uniqid(idrop) = [];

if nargin>6  
    WtW = getMeWtW(W, U);
    cc = max(WtW, [], 3);
    
    cc = cc -diag(diag(cc));
    r0 = 2*abs(mu(:) - mu(:)')./(mu(:) + mu(:)');
    rdir = (nsp(:) - nsp(:)')<0;
    ipair = (cc>0.9 & r0<0.2 & rdir);
    amax = max(ipair, [], 2);
    
%     figure(2)
%     plot(max(cc, [], 2))
%     drawnow
    
    idrop= amax>0;    
    
    W(:,idrop,:) = [];
    U(:,idrop,:) = [];
    dWU(:,:, idrop) = [];
    mu(idrop) = [];
    nsp(idrop) = [];
%     uniqid(idrop) = [];
end
