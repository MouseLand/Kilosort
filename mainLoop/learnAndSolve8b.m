function rez = learnAndSolve8b(rez, iorder)
% This is the main optimization. Takes the longest time and uses the GPU heavily.  

Nbatches = rez.ops.Nbatch;

rng(iseed);
% if getOr(rez.ops, 'midpoint', 0)
%     rez.iorig = randperm(Nbatches);
% end
% rez.istart = ceil(Nbatches/2); % this doesn't really matter anymore


rez.iorig = iorder;

rez     = learnTemplates(rez, rez.iorig);

% if ~isfield(rez, 'W') || isempty(rez.W)    
%     rez     = learnTemplates(rez, rez.iorig);    
% else
%     rez     = learnTemplates2(rez, rez.iorig);
% end

rez.ops.fig = 0;
rez         = runTemplates(rez);
