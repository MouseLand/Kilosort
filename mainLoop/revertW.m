function [W, dWU, sig] = revertW(rez)

W = gpuArray(rez.W);
dWU = gpuArray(rez.dWU);

sig = gpuArray(rez.sig);
