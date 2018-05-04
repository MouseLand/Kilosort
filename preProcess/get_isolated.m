function isol = get_isolated(st0, id0, Mask, nt1)

% nt1 = 41;

imin = 1;
icurrent = 1;
imax = 1;
nspbatch = numel(st0);

isol = true(nspbatch, 1);

nt0 = (size(Mask,1)+1)/2;
while(icurrent<=nspbatch)
    while (st0(imax) - st0(icurrent)) <= nt1-1 && imax<nspbatch
        imax = imax+1;
    end
    
    while (st0(icurrent) - st0(imin)) >= nt1-1
        imin = imin+1;
    end
    for i = [imin:icurrent-1 icurrent+1:imax-1]
        if (Mask(nt0 + (st0(i) - st0(icurrent)), id0(icurrent), id0(i)))
            isol(icurrent) = false;
            break;
        end
    end
    
    icurrent = icurrent + 1;
end
