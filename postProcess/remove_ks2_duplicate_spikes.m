% REMOVE_KS2_DUPLICATE_SPIKES2 Double-counted spikes are hard to avoid with
% Kilosort's template matching algorithm since the overall fit can be
% improved by having multiple templates jointly account for a single variable waveform.
% 
% This function takes the kilosort2 output rez and identifies pair of
% spikes that are close together in time and space. The temporal threshold
% is give by the parameter OVERLAP_S which is 5e-4 (0.5ms by default) and
% the spatial threshold (applied to the template primary sites) is given by
% CHANNEL_SEPARATION_UM and is 50 by default.
%
% From these spike pairs, it identifies the pair with the larger template as
% being the "main" or "reference" cluster and the duplicate spikes from the
% other cluster are removed.
%
% All spike pairs are considered, not just those from CCG-contaminated
% pairs, as in REMOVE_KS2_DUPLICATE_SPIKES2.
%
%=INPUT
%
%   rez structure
%
%=OPTIONAL INPUT, NAME-VALUE PAIRS
%
%   overlap_s
%       the time interval, in second, within which a sequence of spikes are
%       vetted for duplicates.
%
%   channel_separation_um
%       When the primay channels of two spikes are within this distance, in
%       microns, then the two spikes are vetted for duplicate.
%
%=EXAMPLE
%
%   >> rez = remove_ks2_duplicate_spikes(rez)
function rez = remove_ks2_duplicate_spikes(rez, varargin)
    input_parser = inputParser;
    addParameter(input_parser, 'overlap_s', 5e-4, @(x) (isnumeric(x)))
    addParameter(input_parser, 'channel_separation_um', 50, @(x) (ischar(x)))
    parse(input_parser, varargin{:});
    P = input_parser.Results;

    spike_times = uint64(rez.st3(:,1));
    spike_templates = uint32(rez.st3(:,2));

    rez.U=gather(rez.U);
    rez.W = gather(rez.W);
    templates = zeros(rez.ops.Nchan, size(rez.W,1), size(rez.W,2), 'single');
    for iNN = 1:size(templates,3)
       templates(:,:,iNN) = squeeze(rez.U(:,iNN,:)) * squeeze(rez.W(:,iNN,:))';
    end
    templates = permute(templates, [3 2 1]); % now it's nTemplates x nSamples x nChannels

    %% Make sure that the spike times are sorted
    if ~issorted(spike_times)
        [spike_times, I] = sort(spike_times);
        spike_templates = spike_templates(I);
    end
    %% deal with cluster 0
    if any(spike_templates==0)
        error('Currently this function can''t deal with existence of cluster 0. Should be OK since it ought to be run first in the post-processing.');
    end
    %% Determine the channel where each spike had that largest amplitude (i.e., the primary) and determine the template amplitude of each cluster
    whiteningMatrix = rez.Wrot/rez.ops.scaleproc;
    whiteningMatrixInv = whiteningMatrix^-1;

    % here we compute the amplitude of every template...
    % unwhiten all the templates
    tempsUnW = zeros(size(templates));
    for t = 1:size(templates,1)
        tempsUnW(t,:,:) = squeeze(templates(t,:,:))*whiteningMatrixInv;
    end

    % The amplitude on each channel is the positive peak minus the negative
    tempChanAmps = squeeze(max(tempsUnW,[],2))-squeeze(min(tempsUnW,[],2));

    % The template amplitude is the amplitude of its largest channel
    [tempAmpsUnscaled,template_primary] = max(tempChanAmps,[],2);
    %without undoing the whitening
    %template_amplitude = squeeze(max(templates, [], 2) - min(templates, [], 2));
    %[~, template_primary] = max(template_amplitude, [], 2); 

    template_primary = cast(template_primary, class(spike_templates));
    spike_primary = template_primary(spike_templates);

    %% Number of samples in the overlap
    n_samples_overlap = round(P.overlap_s * rez.ops.fs);
    n_samples_overlap = cast(n_samples_overlap, class(spike_times));
    %% Distance between each channel
    chan_dist = ((rez.xcoords - rez.xcoords').^2 + (rez.ycoords - rez.ycoords').^2).^0.5;
    %imagesc(chan_dist)

    n_spikes=numel(spike_times);
    n_duplicates=1;  % set to 1 to initialize while loop
    count=0;
    remove_idx = [];
    reference_idx = [];
    spike_idx = [1:n_spikes]';
    current_spike_times = spike_times;
    current_spike_idx = spike_idx;
    current_primaries = spike_primary;
    % only check nearest temporal neighbors in the list of spikes times.
    % but go recursively until no nearest neighbors are left that are both within the overlap
    % period and sufficiently nearby. 
    % this means only ever computing a vector operation (i.e. diff(spike_times))
    % rather than a matrix one (i.e. spike_times - spike_times').
    diff_order=1;
    while 1==1
        count=count+1;
        if n_duplicates==0
            diff_order=diff_order+1;
            fprintf('No duplicates but simultaneous spikes haven''t been fully explored.\nNow comparing spikes separated by %g other spikes.\n',diff_order-1);
        end
        keep_idx = ~ismember(spike_idx,remove_idx);
        current_spike_idx = spike_idx(keep_idx);
        current_spike_times = spike_times(keep_idx); 
        current_primaries = spike_primary(keep_idx);
        isis=current_spike_times(1+diff_order:end) - current_spike_times(1:end-diff_order);
        simultaneous = isis<n_samples_overlap;
        if any(isis<0)
            error('ISIs less than zero? Something is wrong.');
        end
        if ~any(simultaneous)
            fprintf('No remaining simultaneous spikes.\n');
            break
        end
        nearby = chan_dist(sub2ind(size(chan_dist),current_primaries(1:end-diff_order),current_primaries(1+diff_order:end)))<P.channel_separation_um;
        first_duplicate = find(simultaneous & nearby); % indexes the first member of the pair
        n_duplicates = length(first_duplicate);
        if ~isempty(first_duplicate)
            fprintf('On iteration %g, %g duplicate spike pairs were identified.\n',count,n_duplicates);
            amps_to_compare=tempAmpsUnscaled(current_primaries([first_duplicate first_duplicate(:)+diff_order]));
            if length(first_duplicate)==1
                amps_to_compare = amps_to_compare(:)'; % special case requiring a dimension change
            end
            first_is_bigger =  diff(amps_to_compare,[],2)<=0;
            remove_idx = [remove_idx ; current_spike_idx([first_duplicate(~first_is_bigger);(first_duplicate(first_is_bigger)+diff_order)])];
            reference_idx = [reference_idx ; current_spike_idx([(first_duplicate(~first_is_bigger)+diff_order);first_duplicate(first_is_bigger)])];
            [remove_idx,idx] = unique(remove_idx);
            reference_idx = reference_idx(idx);
        end
    end
    logical_remove_idx = ismember(spike_idx,remove_idx);
    rez = remove_spikes(rez,logical_remove_idx,'duplicate','reference_time',spike_times(reference_idx),...
        'reference_cluster',spike_templates(reference_idx));
end