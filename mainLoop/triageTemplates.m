function [W, U, dWU, mu, nsp, his, flag] = triageTemplates(ops, W, U, dWU, mu, nsp, his, flag)

m0 = ops.minFR * ops.NT/ops.fs;

idrop = nsp<m0;

W(:,idrop,:) = [];
U(:,idrop,:) = [];
dWU(:,:, idrop) = [];
mu(idrop) = [];
nsp(idrop) = [];
his(:, idrop) = [];

flag(idrop) = [];