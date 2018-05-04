function S1 = my_sum(S1, sig, varargin)
% takes an extra argument which specifies which dimension to filter on
% extra argument can be a vector with all dimensions that need to be
% smoothed, in which case sig can also be a vector of different smoothing
% constants

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
    
    S1 = cat(1, 0*ones([sig, dsnew2(2)]),S1, 0*ones([sig, dsnew2(2)]));
    Smax = S1(1:dsnew2(1), :);
    for j = 1:2*sig
        Smax = Smax + S1(j + (1:dsnew2(1)), :);
    end
    
    S1 = reshape(Smax, dsnew);
       
    S1 = permute(S1, [2:idim 1 idim+1:Nd]);
end