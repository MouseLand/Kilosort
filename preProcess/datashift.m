function rez = datashift2(rez)

sig = rez.ops.sig;

ops = rez.ops;
ymin = min(rez.yc);
ymax = max(rez.yc);
xmin = min(rez.xc);

dmin = median(diff(unique(rez.yc)));
fprintf('pitch is %d um\n', dmin)
rez.ops.yup = ymin:dmin/2:ymax; % centers of the upsampled y positions
rez.ops.xup = xmin + [0 16 32 48]; % centers of the upsampled x positions


spkTh = 10; % same as "template amplitude" with generic templates

st3 = standalone_detector(rez, spkTh);
%
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
%
if isfield(ops, 'midpoint')
    [imin1, F1] = align_block(F(:, :, 1:ops.midpoint));
    [imin2, F2] = align_block(F(:, :, ops.midpoint+1:end));
    d0 = align_pairs(F1, F2);
    imin = [imin1 imin2 + d0];
    
else
    switch getOr(ops, 'datashift', 1)
        case 2
            ysamp = dmin + dd * [1:dmax] - dd/2;
            [imin,yblk, F0] = align_block2(F, ysamp);
        case 1
            [imin, F0] = align_block(F);
    end
end

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

dshift = imin * dd;

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



