function S1 = my_min(S1, sig, varargin)
  % returns a running minimum applied sequentially across a choice of dimensions and bin sizes
  % S1 is the matrix to be filtered
  % sig is either a scalar or a sequence of scalars, one for each axis to be filtered.
  %  it's the plus/minus bin length for the minimum filter
  % varargin can be the dimensions to do filtering, if len(sig) != x.shape
  % if sig is scalar and no axes are provided, the default axis is 2


idims = 2;
if ~isempty(varargin)
    idims = varargin{1};
end
if numel(idims)>1 && numel(sig)>1
    sigall = sig;
else
    sigall = repmat(sig, numel(idims), 1);
end

for i = 1:length(idims)
    sig = sigall(i);

    idim = idims(i);
    Nd = ndims(S1);

    S1 = permute(S1, [idim 1:idim-1 idim+1:Nd]);

    dsnew = size(S1);

    S1 = reshape(S1, size(S1,1), []);
    dsnew2 = size(S1);

    S1 = cat(1, Inf*ones([sig, dsnew2(2)]),S1, Inf*ones([sig, dsnew2(2)]));
    Smax = S1(1:dsnew2(1), :);
    for j = 1:2*sig
        Smax = min(Smax, S1(j + (1:dsnew2(1)), :));
    end

    S1 = reshape(Smax, dsnew);

    S1 = permute(S1, [2:idim 1 idim+1:Nd]);
end
