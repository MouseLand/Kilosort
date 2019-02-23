function [rez, X] = splitAllClusters(rez, flag)

ops = rez.ops;
wPCA = gather(ops.wPCA);

ccsplit = rez.ops.ccsplit;

NchanNear   = 32;
Nnearest    = 32;
sigmaMask   = ops.sigmaMask;

ik = 0;
Nfilt = size(rez.W,2);
nsplits= 0;

[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear);

ops.nt0min = getOr(ops, 'nt0min', 20);

[~, iW] = max(abs(rez.dWU(ops.nt0min, :, :)), [], 2);
iW = squeeze(int32(iW));

isplit = 1:Nfilt;
dt = 1/1000;
nccg = 0;

while ik<Nfilt    
    if rem(ik, 100)==1
       fprintf('Found %d splits, checked %d/%d clusters, nccg %d \n', nsplits, ik, Nfilt, nccg) 
    end
    ik = ik+1;
    
    isp = find(rez.st3(:,2)==ik);
    nSpikes = numel(isp);
    if  nSpikes<300
       continue; 
    end
    
    ss = rez.st3(isp,1)/ops.fs;
    
%     [K, Q1] = ccg(ss, ss, 500, dt);    

%     [K, Qi, Q00, Q01, rir] = ccg(ss, ss, 500, dt);
%     Q1 = min(Qi/Q00);
%     R = min(rir);
    
%     if Q1<.2 && R<.05
%         continue;
%     end
    
    clp0 = rez.cProjPC(isp, :, :);
    clp0 = gpuArray(clp0(:,:));    
    clp = clp0 - mean(clp0,1);
    
    
    clp = clp - my_conv2(clp, 250, 1);
    
    if flag
        [u s v] = svdecon(clp');    
        w = u(:,1);
    else
        w = mean(clp0, 1)'; 
        w = w/sum(w.^2)^.5;
    end
    
    x = gather(clp * w);    
    s1 = var(x(x>mean(x)));
    s2 = var(x(x<mean(x)));
    
    mu1 = mean(x(x>mean(x)));
    mu2 = mean(x(x<mean(x)));
    p  = mean(x>mean(x));
    
    logp = zeros(numel(isp), 2);    
    
    for k = 1:50        
        logp(:,1) = -1/2*log(s1) - (x-mu1).^2/(2*s1) + log(p);
        logp(:,2) = -1/2*log(s2) - (x-mu2).^2/(2*s2) + log(1-p);
        
        lMax = max(logp,[],2);
        logp = logp - lMax;
        
        rs = exp(logp);
        
        pval = log(sum(rs,2)) + lMax;
        logP(k) = mean(pval);
        
        rs = rs./sum(rs,2);
        
        p = mean(rs(:,1));
        mu1 = (rs(:,1)' * x )/sum(rs(:,1));
        mu2 = (rs(:,2)' * x )/sum(rs(:,2));
        
        s1 = (rs(:,1)' * (x-mu1).^2 )/sum(rs(:,1));
        s2 = (rs(:,2)' * (x-mu2).^2 )/sum(rs(:,2));
        
        if (k>10 && rem(k,2)==1)
            StS  = clp' * (clp .* (rs(:,1)/s1 + rs(:,2)/s2))/nSpikes;
            StMu = clp' * (rs(:,1)*mu1/s1 + rs(:,2)*mu2/s2)/nSpikes;
            
            w = StMu'/StS;
            w = normc(w');
            x = gather(clp * w);
        end
    end
    
    ilow = rs(:,1)>rs(:,2);
%     ps = mean(rs(:,1));
    plow = mean(rs(ilow,1));
    phigh = mean(rs(~ilow,2));
    nremove = min(mean(ilow), mean(~ilow));

    
    % did this split fix the autocorrelograms?
%     [K, Q12] = ccg(ss(ilow), ss(~ilow), 500, dt);  
    [K, Qi, Q00, Q01, rir] = ccg(ss(ilow), ss(~ilow), 500, dt);
    Q12 = min(Qi/max(Q00, Q01));
    R = min(rir);
    
    % if the CCG has a dip, don't do the split
    if Q12<.25 && R<.05
        nccg = nccg+1;
        continue;
    end
    
    c1  = wPCA * reshape(mean(clp0(ilow,:),1), 3, []);
    c2  = wPCA * reshape(mean(clp0(~ilow,:),1), 3, []);
    cc = corrcoef(c1, c2);
    n1 =sqrt(sum(c1(:).^2));
    n2 =sqrt(sum(c2(:).^2));
    
    r0 = 2*abs(n1 - n2)/(n1 + n2);
    
    
    if cc(1,2)>.9 && r0<.2
        continue;
    end
    
    
    % when do I split 
    if nremove > .05 && min(plow,phigh)>ccsplit && min(sum(ilow), sum(~ilow))>300
       % one cluster stays, one goes
       Nfilt = Nfilt + 1;
       
       rez.dWU(:,iC(:, iW(ik)),Nfilt) = c2;
       rez.dWU(:,iC(:, iW(ik)),ik)    = c1;
       rez.W(:,Nfilt,:) = permute(wPCA, [1 3 2]);
       iW(Nfilt) = iW(ik);
       isplit(Nfilt) = isplit(ik);
       
       rez.st3(isp(ilow), 2)    = Nfilt;
       rez.simScore(:, Nfilt)   = rez.simScore(:, ik);
       rez.simScore(Nfilt, :)   = rez.simScore(ik, :);
       rez.simScore(ik, Nfilt) = 1;
       rez.simScore(Nfilt, ik) = 1;
       
       rez.iNeigh(:, Nfilt)     = rez.iNeigh(:, ik);
       rez.iNeighPC(:, Nfilt)     = rez.iNeighPC(:, ik);
       
       % try this cluster again
       ik = ik-1;
       
       nsplits = nsplits + 1;
       
       X{nsplits} = x;       
    end    
end

fprintf('Finished splitting. Found %d splits, checked %d/%d clusters, nccg %d \n', nsplits, ik, Nfilt, nccg)


Nfilt = size(rez.W,2);
Nrank = 3;
Nchan = ops.Nchan;
Params     = double([0 Nfilt 0 0 size(rez.W,1) Nnearest ...
    Nrank 0 0 Nchan NchanNear 0 0]);

% [rez.W, rez.U, rez.mu] = mexSVDsmall(Params, rez.dWU, rez.W, iC-1, iW-1);
[Ka, Kb] = getKernels(ops, 10, 1);
[rez.W, rez.U, rez.mu] = mexSVDsmall2(Params, rez.dWU, rez.W, iC-1, iW-1, Ka, Kb);

[WtW, iList] = getMeWtW(single(rez.W), single(rez.U), Nnearest);
rez.iList = iList;

isplit = rez.simScore==1;
rez.simScore = gather(max(WtW, [], 3));
rez.simScore(isplit) = 1;

rez.iNeigh   = gather(iList(:, 1:Nfilt));
rez.iNeighPC    = gather(iC(:, iW(1:Nfilt)));

rez.Wphy = cat(1, zeros(1+ops.nt0min, Nfilt, Nrank), rez.W);

rez.isplit = isplit;


% figure(1)
% subplot(1,4,1)
% plot(logP(1:k))
% 
% subplot(1,4,2)
% [~, isort] = sort(x);
% epval = exp(pval);
% epval = epval/sum(epval);
% plot(x(isort), epval(isort))
% 
% subplot(1,4,3)
% ts = linspace(min(x), max(x), 200);
% xbin = hist(x, ts);
% xbin = xbin/sum(xbin);
% 
% plot(ts, xbin)
% 
% figure(2)
% plotmatrix(v(:,1:4), '.')
% 
% drawnow
% 
% % compute scores for splits
% ilow = rs(:,1)>rs(:,2);
% ps = mean(rs(:,1));
% [mean(rs(ilow,1)) mean(rs(~ilow,2)) max(ps, 1-ps) min(mean(ilow), mean(~ilow))]





