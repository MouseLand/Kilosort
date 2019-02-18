function rez = get_ACG(rez)

ops = rez.ops;
dt = 1/1000;

Nk = max(rez.st3(:,2));

% sort by firing rate first

fprintf('initialized spike counts\n')

if ~flag
   rez.R_ACG = [];
   rez.Q_ACG = [];
   rez.K_ACG = [];
end

for j = 1:Nk
    s1 = rez.st3(rez.st3(:,2)==j, 1)/ops.fs;
    
    if numel(s1)==0
        continue;
    end
    [K, Qi, Q00, Q01, rir] = ccg(s1, s1, 500, dt);
    Q = min(Qi/(max(Q00, Q01)));
    R = min(rir);
    
    rez.R_ACG(j) = R;
    rez.Q_ACG(j) = Q;
    rez.K_ACG(:,j) = K;    
   
end