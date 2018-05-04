clear W

tps = [1 5 10 25 40 50 61];
tps = round(tps * nt0/61);
vs  = [0 0  0 -2  1 0   0];
fs= interp1(tps, vs, 1:nt0, 'linear', 'extrap');
W(:,1,1) = my_conv(fs, 2);

tps = [1 5 10 25 40 50 61];
tps = round(tps * nt0/61);
vs  = [0 0  0 -2  1  0  0];
fs= interp1(tps, vs, 1:nt0, 'linear', 'extrap');
W(:,1,2) = my_conv(fs, 2);

tps = [1 5 10 15 25 50 61];
tps = round(tps * nt0/61);
vs  = [0 0  0 -2  1  0  0];
fs= interp1(tps, vs, 1:nt0, 'linear', 'extrap');
W(:,1,3) = my_conv(fs, 2);

tps = [1 5 10 20 30 50 61];
tps = round(tps * nt0/61);
vs  = [0 0  0 -2  1  0  0];
fs= interp1(tps, vs, 1:nt0, 'linear', 'extrap');
W(:,1,4) = my_conv(fs, 2);

tps = [1 5 10 25 40 50 61];
tps = round(tps * nt0/61);
vs  = [0 0  0 -2  0 0   0];
fs= interp1(tps, vs, 1:nt0, 'linear', 'extrap');
W(:,1,5) = my_conv(fs, 2);

tps = [1 5 10 25 40 50 61];
tps = round(tps * nt0/61);
vs  = [0 0  0 -2  0  0  0];
fs= interp1(tps, vs, 1:nt0, 'linear', 'extrap');
W(:,1,6) = my_conv(fs, 2);

tps = [1 5 10 15 25 50 61];
tps = round(tps * nt0/61);
vs  = [0 0  0 -2  0  0  0];
fs= interp1(tps, vs, 1:nt0, 'linear', 'extrap');
W(:,1,7) = my_conv(fs, 2);

tps = [1 5 10 20 30 50 61];
tps = round(tps * nt0/61);
vs  = [0 0  0 -2  0  0  0];
fs= interp1(tps, vs, 1:nt0, 'linear', 'extrap');
W(:,1,8) = my_conv(fs, 2);


W = (single(W));
W = squeeze(W);


W  = W(:, repmat([1 2 3 4 1 2 3 4], ceil(Nfilt/8), 1)) + .1 * my_conv(randn(nt0, 8*ceil(Nfilt/8), 'single')', 5)';
U = repmat(eye(Nchan, Nchan), 1, ceil(Nfilt/Nchan));
U = U(:, 1:Nfilt);
U = my_conv(single(U)', 2)';
U = U .* (1 + .05 * randn(size(U)));
U(abs(U)<.01) = 0;
U = single(U);


if 1<0
    W    = randn(nt0, Nfilt, 'single');
    U    = randn(Nchan, Nfilt, 'single');

    W = single(my_conv(W', 10)');
    U = single(my_conv(U', 10)');

    W = normc(W);
    U = normc(U);
end

Uinit = normc(U);
Winit = normc(W);
