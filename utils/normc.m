function v = normc(v)

v = v./repmat(sum(v.^2, 1), size(v,1),1).^.5;