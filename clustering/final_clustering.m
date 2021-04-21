function rez1 = final_clustering(rez, tF, st3)

wPCA  = rez.wPCA;
wTEMP  = rez.wTEMP;

iC = rez.iC;
ops = rez.ops;


wroll = [];
tlag = [-2, -1, 1, 2];
for j = 1:length(tlag)
    wroll(:,:,j) = circshift(wPCA, tlag(j), 1)' * wPCA;
end

%% split templates into batches
rmin = 0.6;
nlow = 100;
n0 = 0;
use_CCG = 1;

Nchan = rez.ops.Nchan;
Nk = size(iC,2);
yunq = unique(rez.yc);

ktid = int32(st3(:,2));

uweigh = abs(rez.U(:,:,1));
uweigh = uweigh ./ sum(uweigh,1);
ycup = sum(uweigh .* rez.yc, 1);
xcup = sum(uweigh .* rez.xc, 1);

Nfilt =  size(rez.W,2);
dWU = gpuArray.zeros(ops.nt0, ops.Nchan, Nfilt, 'double');
for j = 1:Nfilt
    dWU(:,:,j) = rez.mu(j) * squeeze(rez.W(:, j, :)) * squeeze(rez.U(:, j, :))';
end


ops = rez.ops;
NchanNear   = min(ops.Nchan, 16);

[iC, mask, C2C] = getClosestChannels(rez, ops.sigmaMask, NchanNear);


[~, iW] = max(abs(dWU(ops.nt0min, :, :)), [], 2);
iW = int32(squeeze(iW));

iC = gather(iC(:,iW));
%%
ss = double(st3(:,1)) / ops.fs;

dmin = rez.ops.dmin;
ycenter = (min(rez.yc) + dmin-1):(2*dmin):(max(rez.yc)+dmin+1);
dminx = rez.ops.dminx;
xcenter = (min(rez.xc) + dminx-1):(2*dminx):(max(rez.xc)+dminx+1);
[xcenter, ycenter] = meshgrid(xcenter, ycenter);
xcenter = xcenter(:);
ycenter = ycenter(:);

Wpca = zeros(6, Nchan, 1000, 'single');
nst = numel(ktid);
hid = zeros(nst,1 , 'int32');

xy = zeros(nst, 2);

tic
for j = 1:numel(ycenter)
    if rem(j,5)==1
        fprintf('time %2.2f, GROUP %d/%d, units %d \n', toc, j, numel(ycenter), n0)    
    end
    y0 = ycenter(j);
    x0 = xcenter(j);    
    xchan = (abs(ycup - y0) < dmin) & (abs(xcup - x0) < dminx);
    
    itemp = find(xchan);
        
    tin = ismember(ktid, itemp);
    
    if sum(tin)<1
        continue;
    end
    
    pid = ktid(tin);
    data = tF(tin, :, :);
    
    ich = unique(iC(:, itemp));
%     ch_min = ich(1)-1;
%     ch_max = ich(end);
    
    nsp = size(data,1);
    dd = zeros(nsp, 6,  numel(ich),  'single');
    for k = 1:length(itemp)
        ix = pid==itemp(k);
        [~,ia,ib] = intersect(iC(:,itemp(k)), ich);
        dd(ix, :, ib) = data(ix,:,ia);
    end
    xy(tin, :) =  spike_position(dd, wPCA, wTEMP, rez.xc(ich), rez.yc(ich));

    kid = run_pursuit(dd, nlow, rmin, n0, wroll, ss(tin), use_CCG);
    
    [~, ~, kid] = unique(kid);
    nmax = max(kid);
    for t = 1:nmax
        Wpca(:, ich, t + n0) = gather(sq(mean(dd(kid==t,:,:),1)));
    end
    
    hid(tin) = gather(kid + n0);
    n0 = n0 + nmax;
end
Wpca = Wpca(:,:,1:n0);
toc
%%

rez.xy = xy;

clust_good = check_clusters(hid, ss, .2);
sum(clust_good)

% waveform length was hardcoded at 61. Should be parametric, w/min index for consistent trough polarity
rez.W = zeros(ops.nt0,   0,3, 'single');
rez.U = zeros(ops.Nchan, 0,3, 'single');
rez.mu = zeros(1,0, 'single');
for  t = 1:n0
    dWU = wPCA * gpuArray(Wpca(:,:,t));
    [w,s,u] = svdecon(dWU);
    wsign = -sign(w(ops.nt0min+1, 1));
    rez.W(:,t,:) = wsign * w(:,1:3);
    rez.U(:,t,:) = wsign * u(:,1:3) * s(1:3,1:3);
    rez.mu(t) = sum(sum(rez.U(:,t,:).^2))^.5;
    rez.U(:,t,:) = rez.U(:,t,:) / rez.mu(t);
end

%%
amps = sq(sum(sum(tF.^2,2),3)).^.5;


rez1 = rez;
rez1.st3 = st3;
rez1.st3(:,1) = rez1.st3(:,1); % +  ops.ntbuff;
rez1.st3(:,2) = hid;
rez1.st3(:,3) = amps;


rez1.cProj = [];
rez1.iNeigh = [];
rez1.cProjPC = [];
rez1.iNeighPC = [];
% rez1.st3(:,1) = rez1.st3(:,1); % - 30000 * 2000;

rez1.U = permute(Wpca, [2,3,1]);

rez1.mu = sum(sum(rez1.U.^2, 1),3).^.5;
rez1.U = rez1.U ./ rez1.mu;
rez1.mu = rez1.mu(:);

rez1.W = reshape(rez.wPCA, [ops.nt0, 1, 6]);
rez1.W = repmat(rez1.W, [1, n0, 1]);
rez1.est_contam_rate = ones(n0,1);

