
Nk = max(rez.st3(:,2));

dt = 1/1000;

Q  = ones(Nk,1);
R  = ones(Nk,1);
for i = 1:Nk
    ss = rez.st3(rez.st3(:,2)==i, 1)/ops.fs;
    if numel(ss)==0
        continue;
    end
    [K, Qi, Q00, Q01, rir] = ccg(ss, ss, 500, dt);
    Q(i) = min(Qi/(max(Q00, Q01)));
    R(i) = min(rir);
    
    fr(i) = numel(ss);
end
sum(Q<.2 & R<.05)
%%
fileID = fopen('G:\Spikes\Waksman/ZNP1/cluster_group.tsv','w');

fprintf(fileID, 'cluster_id%sgroup', char(9));
fprintf(fileID, char([13 10]));

for j = 1:length(Q)
    if Q(j)<.2 && R(j)<.05
        fprintf(fileID, '%d%sgood', j-1, char(9));
        fprintf(fileID, char([13 10]));
    end
end
fclose(fileID)

%%
i = 1+433;
j = 1+233;

ss1 = rez.st3(rez.st3(:,2)==i, 1)/ops.fs;
ss2 = rez.st3(rez.st3(:,2)==j, 1)/ops.fs;
[K, Qi ,Q00,Q01, rir] = ccg(ss1, ss2, 500, dt, 0);
K(501) = 0;

Qin = Qi/max(Q00,Q01);

disp(Qin)
disp(rir)

clf
plot(K)


%%
Xsim = rez.simScore;
Nk = size(Xsim,1);
Xsim = Xsim - diag(diag(Xsim));

R = Inf * ones(Nk,Nk);
Q = Inf * ones(Nk,Nk);
for j = 1:Nk
    s1 = rez.st3(rez.st3(:,2)==j, 1)/ops.fs;
    
    ix = find(Xsim(j,:)>.5 & [1:Nk]>j);
    for k = 1:length(ix)        
        s2 = rez.st3(rez.st3(:,2)==ix(k), 1)/ops.fs;
        [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt, 1);
        Q(j,ix(k)) = min(Qi/(max(Q00, Q01)));
        R(j,ix(k)) = min(rir);
    end    
    if rem(j,20)==1
       disp(j) 
    end
end

Q = min(Q, Q');
R = min(R, R');

%%

isum = sum([R<.05 & Q<.2], 2);
plot(isum)
%%
xmatch = R<.05 & Q<.2;
find(xmatch(240,:))    