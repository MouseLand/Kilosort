function rez = remove_lowamp_clusters(rez)

    threshold = 30; % uVpp
    amplitudes = rez.st3(:,3);
spikeTemplates = uint32(rez.st3(:,2));




    templates = gpuArray.zeros(rez.ops.Nchan, size(rez.W,1), size(rez.W,2), 'single');
    for iNN = 1:size(templates,3)
       templates(:,:,iNN) = squeeze(rez.U(:,iNN,:)) * squeeze(rez.W(:,iNN,:))';
    end
    templates = permute(templates, [3 2 1]); % now it's nTemplates x nSamples x nChannels


    whiteningMatrix = rez.Wrot/rez.ops.scaleproc;
    whiteningMatrixInv = whiteningMatrix^-1;

    % here we compute the amplitude of every template...

    % unwhiten all the templates
    tempsUnW = gpuArray.zeros(size(templates));
    for t = 1:size(templates,1)
        tempsUnW(t,:,:) = squeeze(templates(t,:,:))*whiteningMatrixInv;
    end

    % The amplitude on each channel is the positive peak minus the negative
    tempChanAmps = squeeze(max(tempsUnW,[],2))-squeeze(min(tempsUnW,[],2));

    % The template amplitude is the amplitude of its largest channel
    tempAmpsUnscaled = max(tempChanAmps,[],2);

    % assign all spikes the amplitude of their template multiplied by their
    % scaling amplitudes
    spikeAmps = tempAmpsUnscaled(spikeTemplates).*amplitudes;

    % take the average of all spike amps to get actual template amps (since
    % tempScalingAmps are equal mean for all templates)
    ta = clusterAverage(spikeTemplates, spikeAmps);
    tids = unique(spikeTemplates);
    tempAmps(tids) = ta; % because ta only has entries for templates that had at least one spike
    gain = getOr(rez.ops, 'gain', 1);
    tempAmps = gain*tempAmps';
    
    below_threshold = tempAmps<threshold;
    
    to_remove = ismember(spikeTemplates,find(below_threshold));
    
    rez = remove_spikes(rez,to_remove,'amp_threshold');

end