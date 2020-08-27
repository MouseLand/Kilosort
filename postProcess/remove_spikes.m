function rez = remove_spikes(rez,remove_idx,label,varargin)
    if ~islogical(remove_idx)
        error('remove_idx must be logical.');
    end
    if ~any(remove_idx)
        return
    end
    fprintf('Removing %g spikes from rez structure.\n',sum(remove_idx));            
    if ~isfield(rez,'removed')
        [rez.removed.cProj,rez.removed.cProjPC,rez.removed.st2,rez.removed.st3] = deal([]);
        rez.removed.label={};
    end
    L2 = size(rez.removed.cProj,1);
    if ~isempty(rez.cProj)
        rez.removed.cProj = cat(1,rez.removed.cProj,rez.cProj(remove_idx,:));
        rez.removed.cProjPC = cat(1,rez.removed.cProjPC,rez.cProjPC(remove_idx,:,:));
    end
    rez.removed.st3 = cat(1,rez.removed.st3,rez.st3(remove_idx,:));
    rez.removed.st2 = cat(1,rez.removed.st2,rez.st2(remove_idx,:));

    rez.removed.label = cat(1,rez.removed.label,repmat({label},sum(remove_idx),1));
    k=0;
    while k<length(varargin)
        k=k+1;
        if ~isfield(rez.removed,varargin{k}) || size(rez.removed.(varargin{k}),1)<L2
            if isfield(rez.removed,varargin{k})
                L = size(rez.removed.(varargin{k}),1);
            else
                L=0;
            end
            if ischar(varargin{k+1})
                fill='';
                rez.removed.(varargin{k}){L+1:size(rez.removed.cProj,1),1} = fill;
            elseif isnumeric(varargin{k+1})
                fill=NaN;
                rez.removed.(varargin{k})(L+1:L2,1) = fill;
            end
        end   
        if size(varargin{k+1},1)~=sum(remove_idx)
            error('optional arg in incorrect number of rows.');
        end
        rez.removed.(varargin{k}) = cat(1,rez.removed.(varargin{k}),varargin{k+1});
        k=k+1;
    end
    
    if ~isempty(rez.cProj)
        rez.cProj = rez.cProj(~remove_idx,:);
        rez.cProjPC = rez.cProjPC(~remove_idx,:,:);
    end
    rez.st3 = rez.st3(~remove_idx,:);
    rez.st2 = rez.st2(~remove_idx,:);
end