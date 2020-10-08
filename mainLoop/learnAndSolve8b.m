function rez = learnAndSolve8b(rez, iseed)
% This is the main optimization. Takes the longest time and uses the GPU heavily.  


if ~isfield(rez, 'W') || isempty(rez.W)    
    Nbatches = rez.ops.Nbatch;

    rng(iseed);
    if getOr(rez.ops, 'midpoint', 0)
        rez.iorig = randperm(Nbatches);
    end
    rez.istart = ceil(Nbatches/2); % this doesn't really matter anymore        
    
%     if getOr(rez.ops, 'datashift', 0)
%         rng(iseed);
%         rez.iorig = randperm(Nbatches);
%         iorder0 = rez.iorig; %, randperm(Nbatches), randperm(Nbatches)];
%         rez.istart = ceil(Nbatches/2); % this doesn't really matter anymore
%     else
%         starts = [.5, .475, .525, .45, .55];        
%         ihalf = ceil(Nbatches * starts(ceil(iseed/2))); % more robust to start the tracking in the middle of the re-ordered batches
%         
%         % we learn the templates by going back and forth through some of the data,
%         % in the order specified by iorig (determined by batch reordering).
%         % standard order -- learn templates from first half of data starting
%         % from midpoint, counting down to 1, and then returning.
%         
%         iorder0 = rez.iorig([Nbatches:-1:ihalf 1:ihalf]);
%         if rem(iseed,2)==0
%             iorder0 = rez.iorig([1:ihalf Nbatches:-1:ihalf]); % these are absolute batch ids
%         end
%         rez.istart  = rez.iorig(ihalf); % this is the absolute batch id where we start sorting
%     end
    
    rez     = learnTemplates(rez, rez.iorig);
end

rez.ops.fig = 0;
rez         = runTemplates(rez);
