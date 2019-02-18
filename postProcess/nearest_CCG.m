function rez = nearest_CCG(rez)

ops = rez.ops;
dt = 1/1000;

Nk = max(rez.st3(:,2));

% sort by firing rate first

fprintf('initialized spike counts\n')

rez.R0_CCG = [];
rez.Q0_CCG = [];
rez.K0_CCG = [];


Xsim = rez.simScore;
Xsim = Xsim - diag(diag(Xsim));
[~, isort] = sort(Xsim, 1, 'descend'); 

for j = 1:Nk
    s1 = rez.st3(rez.st3(:,2)==j, 1)/ops.fs;
    
    if numel(s1)==0
        continue;
    end
    
    ineigh = isort(1,j);
    s2 = rez.st3(rez.st3(:,2)==ineigh, 1)/ops.fs;
    [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt);
    Q = min(Qi/(max(Q00, Q01)));
    R = min(rir);
    
    rez.R0_CCG(j) = R;
    rez.Q0_CCG(j) = Q;
    rez.K0_CCG(:,j) = K;    
   
end