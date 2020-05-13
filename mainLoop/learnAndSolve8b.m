function rez = learnAndSolve8b(rez)
% This is the main optimization. Takes the longest time and uses the GPU heavily.  


if ~isfield(rez, 'W') || isempty(rez.W)
    Nbatches = numel(rez.iorig);
    ihalf = ceil(Nbatches/2); % more robust to start the tracking in the middle of the re-ordered batches
    
    % we learn the templates by going back and forth through some of the data,
    % in the order specified by iorig (determined by batch reordering).
    % standard order -- learn templates from first half of data starting
    % from midpoint, counting down to 1, and then returning.
    iorder0 = rez.iorig([ihalf:-1:1 1:ihalf]); % these are absolute batch ids
    
    rez     = learnTemplates(rez, iorder0);
    
    rez.istart  = rez.iorig(ihalf); % this is the absolute batch id where we start sorting

else
    rez.WA = [];
end

rez.ops.fig = 0;
rez         = runTemplates(rez);
