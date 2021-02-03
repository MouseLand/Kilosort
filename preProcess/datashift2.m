function rez = datashift2(rez, do_correction)

NrankPC = 6;
[wTEMP, wPCA]    = extractTemplatesfromSnippets(rez, NrankPC);
rez.wTEMP = gather(wTEMP);
rez.wPCA  = gather(wPCA);

if  getOr(rez.ops, 'nblocks', 1)==0
    rez.iorig = 1:rez.temp.Nbatch;
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
% rez.ops.xup = linspace(xmin-16, xmax+16, npt+3); % centers of the upsampled x positions
rez.ops.xup = xmin + [0 16 32 48];

% binning width across Y (um)
dd = 5;
% min and max for the range of depths
dmin = ymin - 1;
dmax  = 1 + ceil((ymax-dmin)/dd);
disp(dmax)


spkTh = 10; % same as the usual "template amplitude", but for the generic templates

% Extract all the spikes across the recording that are captured by the
% generic templates. Very few real spikes are missed in this way. 
[st3, rez] = standalone_detector(rez, spkTh);
%%

% detected depths
% dep = st3(:,2);
% dep = dep - dmin;

Nbatches      = rez.temp.Nbatch;
% which batch each spike is coming from
batch_id = st3(:,5); %ceil(st3(:,1)/dt);

% preallocate matrix of counts with 20 bins, spaced logarithmically
F = zeros(dmax, 20, Nbatches);
for t = 1:Nbatches
    % find spikes in this batch
    ix = find(batch_id==t);
    
    % subtract offset
    dep = st3(ix,2) - dmin;
    
    % amplitude bin relative to the minimum possible value
    amp = log10(min(99, st3(ix,3))) - log10(spkTh);
    
    % normalization by maximum possible value
    amp = amp / (log10(100) - log10(spkTh));
    
    % multiply by 20 to distribute a [0,1] variable into 20 bins
    % sparse is very useful here to do this binning quickly
    M = sparse(ceil(dep/dd), ceil(1e-5 + amp * 20), ones(numel(ix), 1), dmax, 20);    
    
    % the counts themselves are taken on a logarithmic scale (some neurons
    % fire too much!)
    F(:, :, t) = log2(1+M);
end

%%
% determine registration offsets
ysamp = dmin + dd * [1:dmax] - dd/2;
[imin,yblk, F0, F0m] = align_block2(F, ysamp, ops.nblocks);

if isfield(rez, 'F0')
    d0 = align_pairs(rez.F0, F0);
    % concatenate the shifts
    imin = imin - d0;
end

%%
if getOr(ops, 'fig', 1)  
    figure;
    set(gcf, 'Color', 'w')
    
    % plot the shift trace in um
    plot(imin * dd)
    box off
    xlabel('batch number')
    ylabel('drift (um)')
    title('Estimated drift traces')
    drawnow
    
    figure;
    set(gcf, 'Color', 'w')
    % raster plot of all spikes at their original depths
    st_shift = st3(:,2); %+ imin(batch_id)' * dd;
    for j = spkTh:100
        % for each amplitude bin, plot all the spikes of that size in the
        % same shade of gray
        ix = st3(:, 3)==j; % the amplitudes are rounded to integers
        plot(st3(ix, 1)/ops.fs, st_shift(ix), '.', 'color', [1 1 1] * max(0, 1-j/40)) % the marker color here has been carefully tuned
        hold on
    end
    axis tight
    box off

    xlabel('time (sec)')
    ylabel('spike position (um)')
    title('Drift map')
    
end
%%
% convert to um 
dshift = imin * dd;

% this is not really used any more, should get taken out eventually
[~, rez.iorig] = sort(mean(dshift, 2));

if do_correction
    % sigma for the Gaussian process smoothing
    sig = rez.ops.sig;
    % register the data batch by batch
    dprev = gpuArray.zeros(ops.ntbuff,ops.Nchan, 'single');
    for ibatch = 1:Nbatches
        dprev = shift_batch_on_disk2(rez, ibatch, dshift(ibatch, :), yblk, sig, dprev);
    end
    fprintf('time %2.2f, Shifted up/down %d batches. \n', toc, Nbatches)
else
    fprintf('time %2.2f, Skipped shifting %d batches. \n', toc, Nbatches)
end
% keep track of dshift 
rez.dshift = dshift;
% keep track of original spikes
rez.st0 = st3;

rez.F = F;
rez.F0 = F0;
rez.F0m = F0m;

% next, we can just run a normal spike sorter, like Kilosort1, and forget about the transformation that has happened in here 

%%



