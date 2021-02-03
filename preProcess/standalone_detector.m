function [st3, rez] = standalone_detector(rez, spkTh)
% Detects spikes across the entire recording using generic templates.
% Each generic template has rank one (in space-time).
% In time, we use the 1D template prototypes found in wTEMP. 
% In space, we use Gaussian weights of several sizes, centered 
% on (x,y) positions that are part of a super-resolution grid covering the
% entire probe (pre-specified in the calling function). 
% In total, there ~100x more generic templates than channels. 

NrankPC = size(rez.wPCA, 2);
ops = rez.ops;

% minimum/base sigma for the Gaussian. 
sig = 10;

% grid of centers for the generic tempates
[ycup, xcup] = meshgrid(ops.yup, ops.xup);

% determine prototypical timecourses by clustering of simple threshold crossings. 
wTEMP = gpuArray(rez.wTEMP);
wPCA = gpuArray(rez.wPCA);

% Get nearest channels for every template center. 
% Template products will only be computed on these channels. 
NchanNear = 10;
[iC, dist] = getClosestChannels2(ycup, xcup, rez.yc, rez.xc, NchanNear);

% Templates with centers that are far from an active site are discarded
dNearActiveSite = median(diff(unique(rez.yc)));
igood = dist(1,:)<dNearActiveSite;
iC = iC(:, igood);
dist = dist(:, igood);
ycup = ycup(igood);
xcup = xcup(igood);

% number of nearby templates to compare for local template maximum
NchanNearUp =  min(numel(xcup), 10*NchanNear);
[iC2, dist2] = getClosestChannels2(ycup, xcup, ycup, xcup, NchanNearUp);

% pregenerate the Gaussian weights used for spatial components 
nsizes = 5;
v2 = gpuArray.zeros(nsizes, size(dist,2), 'single');
for k = 1:nsizes
    v2(k, :) = sum(exp( - 2 * dist.^2 / (sig * k)^2), 1);
end

% build up Params
NchanUp = size(iC,2);

% preallocate the results
st3 = zeros(1000000, 6);
t0 = 0; %ceil(rez.ops.trange(1) * ops.fs); % I think this should be 0 all the time. 
nsp = 0; % counter for total number of spikes
%%
tic
for k = 1:ops.Nbatch
    % get a batch of whitened and filtered data
    dataRAW = get_batch(rez.ops, k);

    Params = [size(dataRAW,1) ops.Nchan ops.nt0 NchanNear NrankPC ops.nt0min spkTh NchanUp NchanNearUp sig];

    % run the CUDA function on this batch
    [dat, kkmax, st, cF] = spikedetector3(Params, dataRAW, wTEMP, iC-1, dist, v2, iC2-1, dist2);
    
    % upsample the y position using the center of mass of template products
    % coming out of the CUDA function. 
    ys = rez.yc(iC);
    cF0 = max(0, cF);
    cF0 = cF0 ./ sum(cF0, 1);
    iChan = st(2, :) + 1;
    yct = sum(cF0 .* ys(:, iChan), 1);
    
    % build st for the current batch
    st = double(gather(st));
    st(6, :) = st(2,:);
    st(2,:) = gather(yct);
    
    toff = ops.nt0min + t0 + ops.NT*(k-1);
    st(1,:) = st(1,:) + toff; % these offsets ensure the times are computed correctly
    
    st(5,:) = k; % batch number
    
    nsp0 = size(st,2);
    if nsp0 + nsp > size(st3,1)
       st3(nsp + 1e6, 1) = 0; % if we need to preallocate more space
    end
    
    st3(nsp + [1:nsp0], :) = st';
    nsp = nsp + nsp0;
    
    if rem(k,100)==1 || k==ops.Nbatch
        fprintf('%2.2f sec, %d batches, %d spikes \n', toc, k, nsp)
    end
end

st3 = st3(1:nsp, :);
%%

rez.iC = gather(iC);
rez.dist =  gather(dist);

