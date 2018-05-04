function W = alignW(W, ops)

[nt0 , Nfilt] = size(W);


[~, imax] = min(W, [], 1);
dmax = -(imax - ops.nt0min);
% dmax = min(1, abs(dmax)) .* sign(dmax);
 
for i = 1:Nfilt
    if dmax(i)>0
        W((dmax(i) + 1):nt0, i) = W(1:nt0-dmax(i), i);
    else
        W(1:nt0+dmax(i), i) = W((1-dmax(i)):nt0, i);
    end
end




