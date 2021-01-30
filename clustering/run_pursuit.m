function kid = run_pursuit(data, nlow, rmin, n0, wroll, ss, use_CCG)

Xd = gpuArray(data(:, :));
amps = sum(Xd.^2, 2).^.5;

kid = zeros(size(Xd,1), 1);

aj = zeros(1000,1);
for j = 1:1000
    ind = find(kid==0);
%     fprintf('cluster %d\n', n0+j)
    [ix, xold, xnew] = break_a_cluster(Xd(ind, :), wroll, ss(ind), nlow, rmin, use_CCG);
    
    aj(j) = gather(mean(amps(ind(ix))));
%     fprintf('amps = %2.2f \n\n', aj(j));
    kid(ind(ix)) = j;
 
    if length(ix) == length(ind)
        break;
    end
    
end
aj = aj(1:j);

end


function [ix, xold, x] = break_a_cluster(data,wroll,  ss, nlow, rmin, use_CCG)
ix = 1:size(data,1);

xold = [];
dt = 1/1000;
for j = 1:10
    dd = data(ix, :);
    if length(ix) < 2 * nlow
        x = [];
%         disp('done with this cluster (too small)')
        break;
    end
    
    [x, iclust, flag] = bimodal_pursuit(dd, wroll, ss(ix), rmin, nlow, 1, use_CCG);
    
    if flag==0
%        disp('done with this cluster')
       break;
    end

    ix = ix(iclust);
    xold = x;
end    

end



function dd = grab_data(rez, y0, ktid)
xchan = abs(rez.yc - y0) < 20;
itemp = find(xchan(tmp_chan));


tin = ismember(ktid, itemp);
pid = ktid(tin);
data = rez.cProjPC(tin, :, :);

iC = rez.iNeighPC(1:16, :);

ich = unique(iC(:, itemp));
ch_min = ich(1)-1;
ch_max = ich(end);

nsp = size(data,1);
dd = zeros(nsp, 6, ch_max-ch_min, 'single');
for j = 1:length(itemp)
    ix = pid==itemp(j);
    dd(ix, :, iC(:,itemp(j))-ch_min) = data(ix,:,:);
end
dd = dd(:,:);

end
