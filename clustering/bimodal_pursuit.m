function [x, iclust, flag] = bimodal_pursuit(Xd, wroll, ss, rmin, nlow, retry, use_CCG)

dt = 1/1000;
clp = Xd;

mu_clp =  mean(clp,1);
clp = clp - mu_clp;


CC = (clp' * clp)/size(clp,1);
[u,s,v] = svd(CC);
clp = clp * u; % * s(ipc, ipc);
preg = 10/size(clp,1);
nW = preg * 1 + (1 - preg) * mean(clp.^2,1).^.5;
% nW(:) = mean(nW(:));
clp = clp ./ nW;

[nsamp, nk] = size(clp);

% split direction
w = nonrandom_projection(clp);
x = clp * w;
[r, scmax, p, m0, mu1, mu2, sig] = find_split(x);

% disp(r)

logp = gpuArray.zeros(nsamp, 2, 'single');
niter = 50;
logP = gpuArray.zeros(niter,1, 'single');


for k = 1:niter
   logp(:, 1)  = - 1/2 * log(1e-10 + sig) - (x-mu1).^2 / (2*sig) + log(1e-10 + p);
   logp(:, 2)  = - 1/2 * log(1e-10 + sig) - (x-mu2).^2 / (2*sig) + log(1e-10 + 1-p);
   
   lMax = max(logp, [], 2);    
   logp = logp - lMax;
   rs = exp(logp);
   pval = log(sum(rs,2)) + lMax;
   logP(k) = mean(pval);
   rs = rs ./ (1e-10 + sum(rs,2));
   
   if k<niter/2
      [r, scmax, p, m0, mu1, mu2, sig]  = find_split(x); 
   else
      p = mean(rs(:,1));
      mu1 = (rs(:,1)' * x) / (1e-10 + sum(rs(:,1)));
      mu2 = (rs(:,2)' * x) / (1e-10 + sum(rs(:,2)));
      sig = (rs(:,1)' * (x-mu1).^2 + rs(:,2)' * (x-mu2).^2)/nsamp;
   end
   
   StMu = ((mu1 * rs(:,1) + mu2 * rs(:,2))' * clp)/nsamp ;
   w = StMu; % * StSi;
   nww = sum(w.^2)^.5;
   w = w / nww;
   x = clp * w';
end

[rr, scmax, p, m0]  = find_split(x);
w = (w .* nW) * u';
wav1 = mu_clp + mu1 * w;
wav2 = mu_clp + mu2 * w;

m1 = sum(wav1.^2)^.5;
m2 = sum(wav2.^2)^.5;

if m2 > m1
   rs = rs(:, [2,1]);
   rr = rr([2,1]);
   mux = mu2;
   mu2 = mu1;
   mu1 = mux;      
end

n1 = sum(rs(:,1)>.5);
n2 = sum(rs(:,2)>.5);
nmin = min(n1, n2);
% fprintf('%6.0d, %6.0d, %2.2f, %2.2f, %2.4f \n', n1, n2, rr(1), rr(2), abs(mu1-mu2));

flag = 1;
iclust = rs(:,1)>.5;
if (min(rr)<rmin || nmin < nlow)
    flag = 0;
end

if flag==1
    do_roll = 0;
    r0 = mean((wav1-wav2).^2);
    for j = 1:size(wroll,3)
        
       wav = wroll(:,:,j) * reshape(wav2, [6, numel(wav2)/6]);
       wav = wav(:)';
       
       if j==1 || r0 > mean((wav1-wav).^2)
          wav2_best = wav;
          r0 =  mean((wav1-wav2).^2);
          do_roll = 1;
       end
    end
    wav2 = wav2_best;
    
    rc = sum(wav1 .* wav2)/(m1 * m2);
    dmu = 2 * abs(m1-m2)/(m1+m2);
    if rc>.9 && dmu<.2
        flag = 0;
%         fprintf('veto from similarity r = %2.2f, dmu = %2.2f, roll = %d \n', rc, dmu, do_roll)
    end
end

if use_CCG && (flag==1)
    ss1 = ss(iclust);
    ss2 = ss(~iclust);
    [K, Qi, Q00, Q01, rir] = ccg(ss1, ss2, 500, dt); % compute the cross-correlogram between spikes in the putative new clusters
    Q12 = min(Qi/max(Q00, Q01)); % refractoriness metric 1
    R = min(rir);                % refractoriness metric 2
    if Q12<.25 && R<.05 % if both metrics are below threshold.
%         disp('veto from CCG')
        flag = 0;
    end
end

% veto from alignment here
% compute correlation of centroids


if (flag==0) && (retry>0)
    w = w / sum(w.^2).^.5;
    clp = Xd - (Xd * w') * w;
%     disp('one more try')
    [x, iclust, flag] = bimodal_pursuit(clp, wroll, ss, rmin, nlow, retry-1, use_CCG);
end

% sd = sum(mu_clp.^2)^.5;

end

function u = nonrandom_projection(clp)
npow = 6;
nbase = 2;
u = make_rproj(npow, nbase);
u = u - .5; %mean(u(:));

Xd = clp(:, 1:npow);
ntry = size(u,2);
scmax = gpuArray.zeros(ntry, 1, 'single');
w = gpuArray(single(u));
w = w ./ [1:npow]';
w = w ./ sum(w.^2, 1).^.5;

for j = 1:ntry
   x = Xd * w(:,j) ;
   [r, scmax(j)] = find_split(x);
end

[~, imax] = max(scmax);
u = gpuArray.zeros(size(clp,2), 1, 'single');
u(1:npow) = w(:,imax);
end


function u = make_rproj(npow, nbase)
u = zeros(npow, nbase^(npow-1));
for j = 1:nbase^(npow-1)
    u(:, j) = proj_comb(j-1, npow, nbase);
end
end


function u = proj_comb(k, npow, nbase)
u = zeros(npow, 1);
u(1) = 1;
for j = 2:npow
   u(j) = rem(k, nbase); 
   k = floor(k/nbase);
end
end

