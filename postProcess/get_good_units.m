function igood = get_good_units(rez)

NN = size(rez.dWU,3);
sd = zeros(NN, 1); 
for k = 1:NN
    wav = rez.dWU(:, :, k);
    mwav = sq(sum(wav.^2, 1));
    
%     wav = sq(rez.U(:, k, :));
%     mwav = sq(sum(wav.^2, 2));
    
    mmax = max(mwav);
    mwav(mwav<mmax/10) = 0;
    
    xm = mean(rez.xc(:) .* mwav(:)) / mean(mwav);
    ym = mean(rez.yc(:) .* mwav(:)) / mean(mwav);
    
    ds = sqrt((rez.xc(:) - xm).^2 + (rez.yc(:) - ym).^2);
    sd(k) = gather(mean(ds(:) .* mwav(:))/mean(mwav));
    
end
igood = rez.good & sd<100;
igood = double(igood);