function xscore = mergeAllClusters(rez)

Nfilt = size(rez.W,2);

xscore = zeros(Nfilt);
chckd  = zeros(Nfilt);

nmerges = 0;
for ik = 1:Nfilt
   s0 = rez.st3(:,2) ==ik;
   
   ipair = rez.iNeigh(:, ik);
   
   %%
   for j = 2:length(ipair)
       i2 = ipair(j);
       
       if chckd(ik, i2)
           continue;
       end
       
       j2 = find(rez.iNeigh(:, i2)==ik);
       
       if ~isempty(j2) && rez.simScore(ik, i2)>.5
          s1 = rez.st3(:,2) ==i2;
          
          c1 = rez.cProj(s0, [1 j]);
          c2 = cat(1, c1, rez.cProj(s1, [j2 1]));
          
          xs = c2(:,1) - c2(:,2);
          
          if ik==161 && i2==167
              keyboard;
          end
          
          xscore(i2, ik) = mergeScore(xs);           
          chckd(i2, ik) = 1;
          
          if xscore(i2, ik)>0
              nmerges = nmerges + 1;
          end
       end
   end
   %%
   if rem(ik, 100)==1
       fprintf('Found %d merges, checked %d/%d clusters \n', nmerges, ik, Nfilt) 
   end    
    
end

