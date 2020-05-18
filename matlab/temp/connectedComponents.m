function rez = connectedComponents(rez, xscore)

Nfilt = size(rez.W,2);

xscore = xscore + xscore';

chckd = zeros(Nfilt,1);
for j = 1:Nfilt
    if chckd(j)>0
        continue;
    end
    ix = find(xscore(j, :) > 0);
    
    nmerge = 0;
    nmergenew = numel(ix);
    
    while nmergenew>nmerge
        ix = find(sum(xscore(ix, :),1)>0);
        
        nmerge = nmergenew;
        nmergenew = numel(ix);
    end
    
    chckd(ix) = 1;
    
    rez.st3(rez.st3(:,2)==j, 5) = j;           
    if nmerge>0
       for t = 1:length(ix)
          rez.st3(rez.st3(:,2)==ix(t), 5) = j;           
       end
    end    
    
end