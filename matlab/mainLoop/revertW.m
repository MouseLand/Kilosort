function [W, dWU] = revertW(rez)

% to revert the templates, we just need dWU, and W for initialization of the SVD of dWU
W = gpuArray(rez.W);
dWU = gpuArray(rez.dWU);
