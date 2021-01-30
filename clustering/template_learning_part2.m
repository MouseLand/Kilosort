function rez = final_clustering(rez)

wPCA  = rez.wPCA;
wroll = [];
tlag = [-2, -1, 1, 2];
for j = 1:length(tlag)
    wroll(:,:,j) = circshift(wPCA, tlag(j), 1)' * wPCA;
end


%% split templates into batches
rmin = 0.75;
nlow = 100;
n0 = 0;

Nchan = rez.ops.Nchan;
Nk = size(iC,2);
yunq = unique(rez.yc);

ktid = int32(st3(:,2));

uweigh = abs(rez.U(:,:,1));
uweigh = uweigh ./ sum(uweigh,1);
ycup = sum(uweigh .* rez.yc, 1);

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

ycenter = 25:40:3025;

Wpca = zeros(6, Nchan, 1000, 'single');
nst = numel(ktid);
hid = zeros(nst,1 , 'int32');
amp = zeros(1000,1 , 'single');

% ycup = rez.yc;


tic
for j = 1:numel(ycenter)
    fprintf('GROUP %d/%d \n', j, numel(ycenter))    
    y0 = ycenter(j);
    
    xchan = abs(ycup - y0) < 20;
    itemp = find(xchan);
        
    if isempty(itemp)
        continue;
    end
    tin = ismember(ktid, itemp);
    pid = ktid(tin);
    data = tF(tin, :, :);
    
    
    ich = unique(iC(:, itemp));
    ch_min = ich(1)-1;
    ch_max = ich(end);
    
    nsp = size(data,1);
    dd = zeros(nsp, 6, ch_max-ch_min, 'single');
    for k = 1:length(itemp)
        ix = pid==itemp(k);
        dd(ix, :, iC(:,itemp(k))-ch_min) = data(ix,:,:);
    end

    [kid, aj] = run_pursuit(dd, nlow, rmin, n0, wroll, ss(tin));
    
    nmax = max(kid);
    for t = 1:nmax
        Wpca(:, ch_min+1:ch_max, t + n0) = gather(sq(mean(dd(kid==t,:,:),1)));
    end
    
    hid(tin) = gather(kid + n0);
    amp(n0+[1:nmax]) = aj;
    n0 = n0 + nmax;
end
Wpca = Wpca(:,:,1:n0);
amp = amp(1:n0);
toc
%%
clust_good = check_clusters(hid, ss);
sum(clust_good)


rez.W = zeros(61,0, 3, 'single');
rez.U = zeros(ops.Nchan,0,3, 'single');
rez.mu = zeros(1,0, 'single');
for  t = 1:n0
    dWU = wPCA * gpuArray(Wpca(:,:,t));
    [w,s,u] = svdecon(dWU);
    wsign = -sign(w(21,1));
    rez.W(:,t,:) = wsign * w(:,1:3);
    rez.U(:,t,:) = wsign * u(:,1:3) * s(1:3,1:3);
    rez.mu(t) = sum(sum(rez.U(:,t,:).^2))^.5;
    rez.U(:,t,:) = rez.U(:,t,:) / rez.mu(t);
end

%%
rez1 = rez;
rez1.st3 = st3;
rez1.st3(:,1) = rez1.st3(:,1); % +  ops.ntbuff;
rez1.st3(:,2) = hid;
rez1.st3(:,3) = amps;
rez1.simScore = (U(:,:) * U(:,:)') .* (W(:,:) * W(:,:)')/6;

amps = sq(sum(sum(tF.^2,2),3)).^.5;


rez1.cProj = [];
rez1.iNeigh = [];
rez1.cProjPC = [];
rez1.iNeighPC = [];
% rez1.st3(:,1) = rez1.st3(:,1); % - 30000 * 2000;

rez1.U = permute(Wpca, [2,3,1]);

rez1.mu = sum(sum(rez1.U.^2, 1),3).^.5;
rez1.U = rez1.U ./ rez1.mu;
rez1.mu = rez1.mu(:);

rez1.W = reshape(rez.wPCA, [61, 1, 6]);
rez1.W = repmat(rez1.W, [1, n0, 1]);
rez1.est_contam_rate = ones(n0,1);

U = permute(rez1.U, [2,1,3]);
W = permute(rez1.W, [2,1,3]);
%%

rez1 = find_merges(rez1, wroll, 1);

%%
hid = int32(rez1.st3(:,2));
clust_good = check_clusters(hid, ss);
sum(clust_good)
rez1.good = clust_good;

%%
rezToPhy2(rez1, rootZ);


%%
tmp_chan = iC(1,:);
y0 = 700;
xchan = abs(rez.yc - y0) < 20;
itemp = find(xchan(tmp_chan));

tin = ismember(ktid, itemp);
pid = ktid(tin);
data = rez.cProjPC(tin, :, :);

ich = unique(iC(:, itemp));
ch_min = ich(1)-1;
ch_max = ich(end);

nsp = size(data,1);
dd = zeros(nsp, 6, ch_max-ch_min, 'single');
for k = 1:length(itemp)
    ix = pid==itemp(k);
    dd(ix, :, iC(:,itemp(k))-ch_min) = data(ix,:,:);
end

size(dd)
%%
tic
kid = run_pursuit(dd, nlow, rmin, 0);
toc
%%
subplot(2,1,1)
[~, isort] = sort(pid);
imagesc(dd(isort(1:10:end), :)', [-10, 10])
colormap(redblue)
subplot(2,1,2)
[~, isort] = sort(kid);
imagesc(dd(isort(1:10:end), :)', [-10, 10])

hold on
ksort = sort(kid);
nc = max(kid);
plot(size(dd,2) * ksort(1:10:end) / nc, 'k', 'linewidth', 3)
