function [st3] = standalone_detector(rez, spkTh)

sig = 10;
dNearActiveSite = 30;
ops = rez.ops;

[ycup, xcup] = meshgrid(ops.yup, ops.xup);

NrankPC = 6;
[wTEMP, wPCA]    = extractTemplatesfromSnippets(rez, NrankPC);

NchanNear = 10;
[iC, dist] = getClosestChannels2(ycup, xcup, rez.yc, rez.xc, NchanNear);

igood = dist(1,:)<dNearActiveSite;
iC = iC(:, igood);
dist = dist(:, igood);

ycup = ycup(igood);
xcup = xcup(igood);

NchanNearUp =  10*NchanNear;
[iC2, dist2] = getClosestChannels2(ycup, xcup, ycup, xcup, NchanNearUp);

nsizes = 5;
v2 = gpuArray.zeros(5, size(dist,2), 'single');
for k = 1:nsizes
    v2(k, :) = sum(exp( - 2 * dist.^2 / (sig * k)^2), 1);
end

NchanUp = size(iC,2);

Params = [ops.NT ops.Nchan ops.nt0 NchanNear NrankPC ops.nt0min spkTh NchanUp NchanNearUp sig];

st3 = zeros(1000000, 5);
t0 = ceil(rez.ops.trange(1) * ops.fs);
nsp = 0;

tic
for k = 1:ops.Nbatch
    dataRAW = get_batch(rez.ops, k);
    
    [dat, kkmax, st, cF] = spikedetector3(Params, dataRAW, wTEMP, iC-1, dist, v2, iC2-1, dist2);
    
    % compute y position    
    ys = rez.yc(iC);
    cF0 = max(0, cF);
    cF0 = cF0 ./ sum(cF0, 1);
    iChan = st(2, :) + 1;
    yct = sum(cF0 .* ys(:, iChan), 1);
    
    st = double(gather(st));
    st(2,:) = gather(yct);
    
    ioffset         = ops.ntbuff;
    if k==1
        ioffset         = 0; % the first batch is special (no pre-buffer)
    end
    toff = ops.nt0min + t0 -ioffset + (ops.NT-ops.ntbuff)*(k-1);
    st(1,:) = st(1,:) + toff;
    
    st(5,:) = k;
    
    nsp0 = size(st,2);
    if nsp0 + nsp > size(st3,1)
       st3(nsp + 1e6, 1) = 0;
    end
    
    st3(nsp + [1:nsp0], :) = st';
    nsp = nsp + nsp0;
    
    if rem(k,100)==1 || k==ops.Nbatch
        fprintf('%2.2f sec, %d batches, %d spikes \n', toc, k, nsp)
    end
end

