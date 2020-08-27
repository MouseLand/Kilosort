function rez = remove_inactive_clusters(rez,varargin)

    p=inputParser;
    p.addParameter('min_spikes',50,@(x)validateattributes(x,{'numeric'},{'scalar','nonnegative'}));
    p.parse(varargin{:});
    params=p.Results;
    [template_idx,~,uniqueIdx]=unique(rez.st3(:,2));
    n_spikes=accumarray(uniqueIdx,1);
    n_spikes = n_spikes(template_idx>0);
    template_idx = template_idx(template_idx>0);
    
    below_n_spikes_threshold= n_spikes<params.min_spikes;
    recording_duration_sec = rez.ops.sampsToRead./rez.ops.fs;
    below_rate_threshold = n_spikes<rez.ops.minFR*recording_duration_sec;
    
    remove_clusters = below_n_spikes_threshold | below_rate_threshold;
    
    spikes_to_remove = ismember(rez.st3(:,2),template_idx(remove_clusters));
    
    rez.inactive = template_idx(remove_clusters);
    
    rez = remove_spikes(rez,spikes_to_remove,'inactive');
    rez = recompute_clusters(rez);
    
    fprintf('Removed %g inactive clusters.\n',sum(remove_clusters));

end