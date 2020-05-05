function [st3, imin] = shift_from_hist(rez, flag)

ymin = min(rez.yc);
ymax = max(rez.yc);
xmin = min(rez.xc);

if flag
    rez.ops.yup = ymin:7.5:ymax; % centers of the upsampled y positions
    rez.ops.xup = xmin + [0 16 32]; % centers of the upsampled x positions
else
    rez.ops.yup = ymin:10:ymax; % centers of the upsampled y positions
    rez.ops.xup = [11 27 43 59]; %[0 16 32]; % centers of the upsampled x positions
end

spkTh = 10; % same as "template amplitude" with generic templates

st3 = standalone_detector(rez, spkTh);

dd = 5;
dep = st3(:,2);
dmin = ymin - 1;
dep = dep - dmin;

dmax  = 1 + ceil(max(dep)/dd);
ops = rez.ops;
dt = ops.NT;
NT = max(st3(:,1));
Nbatches = ceil(NT/dt);

batch_id = ceil(st3(:,1)/dt);

F = zeros(dmax, 20, Nbatches);
for t = 1:Nbatches
    ix = find(batch_id==t);
    dep = st3(ix,2) - dmin;
    amp = log10(min(99, st3(ix,3))) - log10(spkTh);
    amp = amp / (log10(100) - log10(spkTh));
    
    M = sparse(ceil(dep/dd), ceil(1e-5 + amp * 20), ones(numel(ix), 1), dmax, 20);    
    F(:, :, t) = log2(1+M);
end

[imin, F0] = align_block(F);

imin = imin * 5;
