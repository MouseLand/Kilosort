function v = normc(v)
% normalize along first axis (works for GPU arrays)
v = v./repmat(sum(v.^2, 1), size(v,1),1).^.5;
