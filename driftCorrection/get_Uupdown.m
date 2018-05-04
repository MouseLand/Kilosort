
function [Uup, Udown] = get_Uupdown(U, space_lag, CovChans, Wrot, ops)

Uwh = (Wrot * (CovChans/Wrot)) * U;


Udown = reshape(U, 2, ops.Nchan/2, [], size(U,2));
Udown(:,2:end,:,:)   = Udown(:,1:end-1,:,:);
Udown = reshape(Udown, size(U));
Udown = normc(Udown);
Uup = reshape(U, 2, ops.Nchan/2, [], size(U,2));
Uup(:,1:end-1,:,:) = Uup(:,2:end,:,:);
Uup = reshape(Uup, size(U));
Uup = normc(Uup);