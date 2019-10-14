

function [values,instances] = countUnique(x)
% function [values,instances] = countUnique(x)

y = sort(x(:));
 p = find([true;diff(y)~=0;true]);
 values = y(p(1:end-1));
 instances = diff(p);