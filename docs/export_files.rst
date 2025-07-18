Files exported for Phy
======================

The following files will be saved in the results directory. Note that 'template'
here does *not* refer to the universal or learned templates used for spike
detection, as it did in some past versions of Kilosort. Instead, it refers
to the average spike waveform (after whitening, filtering, and drift
correction) for all spikes assigned to each cluster, which are template-like
in shape. We use the term 'template' anyway for this section because that is
how they are treated in Phy. Elsewhere in the Kilosort4 code, we would refer
to these as 'clusters.'

amplitudes.npy : shape (n_spikes,)
    Per-spike amplitudes, computed as the L2 norm of the PC features
    for each spike.
channel_map.npy : shape (n_channels,)
    Same as probe['chanMap']. Integer indices into rows of binary file
    that map the data to the contacts listed in the probe file.
channel_positions.npy : shape (n_channels,2)
    Same as probe['xc'] and probe['yc'], but combined in a single array.
    Indicates x- and y- positions (in microns) of probe contacts.
channel_shanks.npy : shape (n_channels,)
    Indicates shank index for each channel on the probe.
cluster_Amplitude.tsv : shape (n_templates,)
    Per-template amplitudes, computed as the L2 norm of the template.
cluster_ContamPct.tsv : shape (n_templates,)
    Contamination rate for each template, computed as fraction of refractory
    period violations relative to expectation based on a Poisson process.
cluster_KSLabel.tsv : shape (n_templates,)
    Label indicating whether each template is 'mua' (multi-unit activity)
    or 'good' (refractory).
cluster_group.tsv : shape (n_templates,)
    Same as `cluster_KSLabel.tsv`.
kept_spikes.npy : shape (n_spikes,)
    Boolean mask that is False for spikes that were removed by
    `kilosort.postprocessing.remove_duplicate_spikes` and True otherwise.
ops.npy : shape N/A
    Dictionary containing a number of state variables saved throughout
    the sorting process (see `run_kilosort`). We recommend loading with
    `kilosort.io.load_ops`.
params.py : shape N/A
    Settings used by Phy, like data location and sampling rate.
pc_features.npy : shape (n_spikes, n_pcs, nearest_chans)
    Temporal features for each spike on the nearest channels for the
    template the spike was assigned to.
pc_feature_ind.npy : shape (n_templates, nearest_chans)
    Channel indices of the nearest channels for each template.
similar_templates.npy : shape (n_templates, n_templates)
    Similarity score between each pair of templates, computed as correlation
    between templates.
spike_clusters.npy : shape (n_spikes,)
    For each spike, integer indicating which template it was assigned to.
spike_templates.npy : shape (n_spikes,2)
    Same as `spike_clusters.npy`.
spike_positions.npy : shape (n_spikes,2)
    Estimated (x,y) position relative to probe geometry, in microns,
    for each spike.
spike_times.npy : shape (n_spikes,)
    Sample index of the waveform peak for each spike.
templates.npy : shape (n_templates, nt, n_channels)
    Full time x channels template shapes.
templates_ind.npy : shape (n_templates, n_channels)
    Channel indices on which each cluster is defined. For KS4, this is always
    all channels, but Phy requires this file.
whitening_mat.npy : shape (n_channels, n_channels)
    Matrix applied to data for whitening.
whitening_mat_inv.npy : shape (n_channels, n_channels)
    Inverse of whitening matrix.
whitening_mat_dat.npy : shape (n_channels, n_channels)
    matrix applied to data for whitening. Currently this is the same as
    `whitening_mat.npy`, but was added because the latter was previously
    altered before saving for Phy, so this ensured the original was still
    saved. It's kept in for now because we may need to change the version
    used by Phy again in the future.