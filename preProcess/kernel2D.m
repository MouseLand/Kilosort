function K = kernel2D(xp, yp, sig)

distx = abs(xp(:, 1) - yp(:,1)');
disty = abs(xp(:, 2) - yp(:,2)');

sigx = sig;
sigy = 1.5 * sig;

p = 1;
K = exp(- (distx/sigx).^p - (disty/sigy).^p);
