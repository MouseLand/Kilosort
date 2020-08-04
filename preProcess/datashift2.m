function rez = datashift2(rez)

if  getOr(rez.ops, 'datashift', 1)==0
    return;
end

ops = rez.ops;

% The min and max of the y and x ranges of the channels
ymin = min(rez.yc);
ymax = max(rez.yc);
xmin = min(rez.xc);
xmax = max(rez.xc);

% Determine the average vertical spacing between channels. 
% Usually all the vertical spacings are the same, i.e. on Neuropixels probes. 
dmin = median(diff(unique(rez.yc)));
fprintf('pitch is %d um\n', dmin)
rez.ops.yup = ymin:dmin/2:ymax; % centers of the upsampled y positions

% Determine the template spacings along the x dimension
xrange = xmax - xmin;
npt = floor(xrange/16); % this would come out as 16um for Neuropixels probes, which aligns with the geometry. 
rez.ops.xup = linspace(xmin, xmax, npt+1); % centers of the upsampled x positions

spkTh = 10; % same as the usual "template amplitude", but for the generic templates

% Extract all the spikes across the recording that are captured by the
% generic templates. Very few real spikes are missed in this way. 
st3 = standalone_detector(rez, spkTh);

dd = 5;
dep = st3(:,2);
dmin = ymin - 1;
dep = dep - dmin;

dmax  = 1 + ceil(max(dep)/dd);
Nbatches      = rez.temp.Nbatch;

batch_id = st3(:,5); %ceil(st3(:,1)/dt);

F = zeros(dmax, 20, Nbatches);
for t = 1:Nbatches
    ix = find(batch_id==t);
    dep = st3(ix,2) - dmin;
    amp = log10(min(99, st3(ix,3))) - log10(spkTh);
    amp = amp / (log10(100) - log10(spkTh));
    
    M = sparse(ceil(dep/dd), ceil(1e-5 + amp * 20), ones(numel(ix), 1), dmax, 20);    
    F(:, :, t) = log2(1+M);
end

if isfield(ops, 'midpoint')
    [imin1, F1] = align_block(F(:, :, 1:ops.midpoint));
    [imin2, F2] = align_block(F(:, :, ops.midpoint+1:end));
    d0 = align_pairs(F1, F2);
    imin = [imin1 imin2 + d0];
    imin = imin - mean(imin);
    ops.datashift = 1;    
else
    switch getOr(ops, 'datashift', 1)
        case 2
            ysamp = dmin + dd * [1:dmax] - dd/2;
            [imin,yblk, F0] = align_block2(F, ysamp);
        case 1
            [imin, F0] = align_block(F);
    end
end

if getOr(ops, 'fig', 1)
    
    figure(193)
    plot(imin * dd)
    drawnow
    
    figure;
    st_shift = st3(:,2); %+ imin(batch_id)' * dd;
    for j = spkTh:100
        ix = st3(:, 3)==j; % the amplitudes are rounded to integers
        plot(st3(ix, 1), st_shift(ix), '.', 'color', [1 1 1] * max(0, 1-j/40)) %, 'markersize', j)
        hold on
    end
    axis tight
end

if ~isempty(getOr(ops, 'fbinaryproc', []))
    fid2 = fopen(ops.fbinaryproc, 'w');
    fclose(fid2);
end

dshift = imin * dd;
[~, rez.iorig] = sort(dshift);

%%
sig = rez.ops.sig;

for ibatch = 1:Nbatches
    switch getOr(ops, 'datashift', 1)
        case 2
            shift_batch_on_disk2(rez, ibatch, dshift(ibatch, :), yblk, sig);
        case 1
            shift_batch_on_disk(rez, ibatch, dshift(ibatch), sig);
    end
end
fprintf('time %2.2f, Shifted up/down %d batches. \n', toc, Nbatches)


rez.dshift = dshift;
rez.st0 = st3;


%%



