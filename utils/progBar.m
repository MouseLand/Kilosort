classdef progBar < dynamicprops
    % Command line text progress bar for indexed loops
    % EXAMPLE
    % Setup:   pb = progBar(allIdx, updatePct);
    %          % allIdx == set of indices used in loop
    %          % updatePct == percentiles to trigger text updates (def=[10:10:90])
    %
    % Use:     for i = allIdx
    %               % do stuff
    %               pb.check(i);
    %          end
    %
    %
    % 2020-07-23 TBC wrote object oriented progress bar class (czuba@utexas.edu)
    
    properties (Access = public)
        pct
        vals
        vals0
        idx
        n
        txt
        d
        lims
    end
    
    methods
        % constructor
        function pb = progBar(vin, pct, varargin)
            % Parse inputs & setup default parameters
            pp = inputParser();
            pp.addParameter('pct',10:10:90); % update points
            pp.parse(varargin{:});
            argin = pp.Results;
           
            % Apply to object
            fn = fieldnames(argin);
            for i = 1:length(fn)
                % add property if non-standard
                if ~isprop(pb, fn{i})
                    pb.addprop(fn{i});
                end
                pb.(fn{i}) = argin.(fn{i});
            end

            if nargin>1 && ~isempty(pct)
                if isscalar(pct)
                    pct = linspace(0,100,pct);
                    pct = pct(2:end-1);
                end
                pb.pct = pct;
            end

            % select update values from [vin]
            pb.lims = vin([1,end]);
            pb.vals = prctile(vin, pb.pct);
            % find nearest vals present
            [~,pb.idx] = min(abs(pb.vals - vin'));
            pb.vals = vin(pb.idx);
            pb.vals0 = pb.vals; % backup init state
            % info
            pb.n = length(pb.vals)+1;
            initialize(pb);
            pb.d = [repmat('\b',1,pb.n+3),'\n'];
            
        end
        
        function initialize(pb)
            pb.vals = pb.vals0;
            pb.txt = char(kron('.|', ones(1,pb.n)));
        end
        
        
        % check for update
        function out = check(pb, i)
            tmp = [];
            hit = false;
            if nargin<2 
                % display text
                tmp = sprintf([,'\n[',pb.txt(1:pb.n),']']);
            elseif i==pb.lims(1)
                % reset text
                initialize(pb);
                % display text
                tmp = sprintf([,'\n[',pb.txt(1:pb.n),']']);

                
            elseif i==pb.vals(1) || i==pb.lims(end)
                %increment vals & text list
                pb.vals = circshift(pb.vals, -1);
                pb.txt = circshift(pb.txt, 1);
                % update text & display
                tmp = sprintf([pb.d,'[',pb.txt(1:pb.n),']']);
                if i==pb.lims(end)
                    tmp = sprintf('%s Done.\n',tmp);
                end
                hit = true;
            end
            
            if ~isempty(tmp)
                fprintf(tmp)
            end
            if nargout>0
                out = hit;
            end

        end
        
        function reset(pb)
            initialize(pb);
        end
    end
end
        

