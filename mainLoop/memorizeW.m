function rez = memorizeW(rez, W, dWU, U, mu)

rez.dWU = dWU;

rez.W = gather(W);
rez.U = gather(U);
rez.mu = gather(mu);