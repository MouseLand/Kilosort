function rez = memorizeW(rez, W, dWU,U,mu, sig)

rez.dWU = dWU;
rez.W = gather(W);
rez.sig = gather(sig);

rez.U = gather(U);
rez.mu = gather(mu);

for n = 1:size(U,2)
    rez.Wraw(:,:,n) = mu(n) * sq(Urot(:,n,:)) * sq(rez.W(:,n,:))';
end
