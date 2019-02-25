function [W, dWU] = revertW(rez)

W = gpuArray(rez.W);
dWU = gpuArray(rez.dWU);