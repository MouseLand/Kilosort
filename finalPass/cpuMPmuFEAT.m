function [sts, ids, xs, Costs, cprojall] = cpuMPmuFEAT(Params,data,fW,WtW, mu, lam1, nu, ops)

nt0 = ops.nt0;

WtW     = permute(WtW, [1 3 2]);

NT      = Params(1);
nFilt   = Params(2);
Th      = Params(3);

fdata   = fft(data, [], 1);
proj    = real(ifft(fdata .* fW(:,:), [], 1));
if ops.Nrank > 1
    proj    = sum(reshape(proj, NT, nFilt, ops.Nrank),3);
end
trange = int32([-(nt0-1):(nt0-1)]);

xs      = zeros(Params(4), 1, 'single');
ids     = zeros(Params(4), 1, 'int32');
sts     = zeros(Params(4), 1, 'int32');
Costs    = zeros(Params(4), 1, 'single');
cprojall = zeros(Params(4), nFilt, 'single');

i0 = 0;
for k = 1:30
    Ci = bsxfun(@plus, proj, (mu.*lam1)');
    Ci = bsxfun(@rdivide, Ci.^2,  1 + lam1');
    Ci = bsxfun(@minus, Ci, (lam1 .* mu.^2)');
    
    [mX, id] = max(Ci,[], 2);
    
    maX         = -my_min(-mX, 31, 1);
    id          = int32(id);
    
    st                   = find((maX < mX + 1e-3) & mX > Th*Th);
    st(st>NT-nt0 | st<nt0) = [];
    
    if isempty(st)
       break; 
    end
    id      = id(st);
    
    % inds = bsxfun(@plus, st', [1:nt0]');
    
    x       = zeros(size(id));
    Cost    = zeros(size(id));
    nsp     = zeros(nFilt,1);
    cproj   = zeros(size(id,1), nFilt, 'single');
    for j = 1:numel(id)
        x(j)            = proj(st(j), id(j));
        Cost(j)         = maX(st(j));
        nsp(id(j))      = nsp(id(j)) + 1;
       
        % subtract off WtW 
        cproj(j,:) =  proj(st(j) ,:);
        proj(st(j) + trange,:) = proj(st(j) + trange,:)  - x(j) * WtW(:,:,id(j));
    end
    
    xs(i0 + [1:numel(st)])          = x;
    sts(i0 + [1:numel(st)])         = st;
    Costs(i0 + [1:numel(st)])       = Cost;
    ids(i0 + [1:numel(st)])         = id;
    cprojall(i0 + [1:numel(st)], :) = cproj;
    i0 = i0 + numel(st);
end


ids     = ids(1:i0);
xs      = xs(1:i0);
Costs   = Costs(1:i0);
sts     = sts(1:i0);
cprojall = cprojall(1:i0, :);
cprojall = cprojall';

ids = ids - 1;



% keyboard
