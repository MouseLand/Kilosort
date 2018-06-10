function rez = memorizeW(rez, W, dWU,U,mu, sig)

rez.dWU = dWU;
rez.W = gather(W);
rez.sig = gather(sig);

rez.U = gather(U);
rez.mu = gather(mu);