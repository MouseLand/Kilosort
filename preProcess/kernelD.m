function Kshift = kernelD(xcoords, ycoords, sig, dshift)

ds = (xcoords(:) - xcoords(:)').^2 + (ycoords(:) - ycoords(:)').^2;
Kdd = exp(-ds/sig^2);

dnew = (xcoords(:) - xcoords(:)').^2 + (ycoords(:) + dshift - ycoords(:)').^2;
Kdx = exp(-dnew/sig^2);

Kshift = Kdx / (Kdd + 1e-2 * eye(size(Kdd,1)));

[amin, imin] = min(dnew, [], 2);


n0 = size(Kshift,1);

Kcopy = zeros(n0);
Kcopy((imin-1)*n0 + [1:n0]') = 1;

Kshift(amin<5^2, :) = Kcopy(amin<3^2, :);



