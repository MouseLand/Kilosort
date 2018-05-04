% assemble dataset

probeName = {'K1', 'K2', 'K3', 'ZNP1', 'ZNP2', 'ZNP3', 'ZNP4', 'ZO'};

rootAlignments = '\\zserver.cortexlab.net\Data\Subjects\';

nt0 = 3e4;
NT = 2135;
    
spks = [];
clu  = [];
st = [];

iMouse = 2;

mname  = {'Waksman',    'Krebs',      'Robbins'};
datexp = {'2017-06-10', '2017-06-05', '2017-06-13'};
rootZ = '\\zserver\Data\Subjects';

Nmax = 0;
for j = 1:length(probeName)
    frootAlign = fullfile(rootAlignments, mname{iMouse}, datexp{iMouse}, 'alignments');
   
    ops.dir_rez     = 'G:\DATA\Spikes\';
    fname = sprintf('correct_ephys_%s_to_ephys_K1.npy', probeName{j});
    if j>1
        boff = readNPY(fullfile(frootAlign, fname));
    else
        boff = [1 0]';
    end
    
    % save the result file
    fname = fullfile(ops.dir_rez,  sprintf('rez_%s_%s_%s.mat', mname{iMouse}, ...
            datexp{iMouse}, probeName{j}));
    load(fname);

    NN = rez.ops.Nfilt;
    
    t0 = ceil(rez.ops.trange(1) * ops.fs); 

    nSpikes = numel(rez.st);
    
    st0 = t0 + rez.st;
    spks(j).st  = [st0(:)/ops.fs ones(nSpikes,1)] * boff;
    
    spks(j).clu = rez.clu(:);
    
    rez.W(7*374, 1) = 0;
    spks(j).W   = reshape(gather(rez.W), 7, 374, []);
    [~, spks(j).Wheights] = max(sq(sum(spks(j).W.^2,1)), [], 1);
    spks(j).wPCA   = rez.wPCA;
    
    ycoords = rez.ycoords(rez.connected>0);
    
    spks(j).Wheights = ycoords(spks(j).Wheights);
    
    clu = cat(1, clu, Nmax + rez.clu(:));
    st  = cat(1, st, spks(j).st(:));
    
    Nmax = Nmax + max(rez.clu);
 end

save(fullfile('G:\DATA\Spikes\', sprintf('spks%s.mat', mname{iMouse})), 'spks')
%%

S = sparse(max(1, ceil(st - rez.ops.trange(1))), clu, ones(1, numel(clu)));

Sall = gpuArray(single(full(S)));
Sall = Sall(15:end-15, :);


%%
Slow = my_conv2(Sall,500,1);
rat = min(Slow, [], 1) ./max(Slow, [],1);
S0 = Sall(:, rat>.5);

[U S V] = svdecon(S0 - mean(S0,1));

plot(U(:,3))

%%

plot(U(:,4))