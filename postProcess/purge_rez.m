function rez = purge_rez(rez)

Nk = size(rez.W, 2);
nsp = zeros(Nk,1);
for j = 1:Nk
   nsp(j) = sum(rez.st3(:,2)==j); 
end
igood = nsp > 10;

imap = cumsum(igood);
rez.st3(:,2) = imap(rez.st3(:,2));

rez.W = rez.W(:, igood, :);
rez.dWU = rez.dWU(:,:,igood);
rez.simScore = rez.simScore(igood, igood); 
rez.iNeighPC = rez.iNeighPC(:, igood);
rez.iNeigh = rez.iNeigh(:,igood);
% rez.Wraw = rez.Wraw(:,:,igood);
rez.nsp = nsp(igood);
rez.good = rez.good(igood);
rez.est_contam_rate = rez.est_contam_rate(igood);
rez.Ths = rez.Ths(igood);
rez.mu = rez.mu(igood);
rez.U = rez.U(:, igood, :);
