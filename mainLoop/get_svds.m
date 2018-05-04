function [W, U, mu] = get_svds(dWU, Nrank)

[Wall, Sv, Uall] = svd(gather_try(dWU), 0);
[~, imax] = max(abs(Wall(:,1)));
Uall(:,1) = -Uall(:,1) * sign(Wall(imax,1));
Wall(:,1) = -Wall(:,1) * sign(Wall(imax,1));

%     [~, imin] = min(diff(Wall(:,1), 1));
%     [~, imin] = min(Wall(:,1));
%     dmax(k) = - (imin- 20);

%     if dmax(k)>0
%         dWU((dmax(k) + 1):nt0, :,k) = dWU(1:nt0-dmax(k),:, k);
%         Wall((dmax(k) + 1):nt0, :)  = Wall(1:nt0-dmax(k),:);
%     else
%         dWU(1:nt0+dmax(k),:, k) = dWU((1-dmax(k)):nt0,:, k);
%         Wall(1:nt0+dmax(k),:) = Wall((1-dmax(k)):nt0,:);
%     end

Wall = Wall * Sv;

Sv = diag(Sv);
mu = sum(Sv(1:Nrank).^2).^.5;
Wall = Wall/mu;

W = Wall(:,1:Nrank);
U = Uall(:,1:Nrank);