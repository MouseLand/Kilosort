from pathlib import Path
import numpy as np
from kilosort import io, CCG


def mean_waveform(cluster_id, results_dir, n_spikes=np.inf, bfile=None, best=True):
    """Get mean waveform for `n_spikes` random spikes assigned to `cluster_id`.

    Parameters
    ----------
    cluster_id : int
        Cluster index to reference from `spike_clusters.npy` in the results
        directory. Only waveforms from spikes assigned to this cluster will
        be used.
    results_dir : str or Path
        Path to directory where Kilosort4 sorting results were saved.
    n_spikes : int; default=np.inf
        Number of spikes to use for computing mean. By default, all spikes
        assigned to `cluster_id` are used.
    bfile : kilosort.io.BinaryFiltered; optional
        Kilosort4 data file object. By default, this will be loaded using the
        information in `ops.npy` in the saved results.
    best : bool; default=True
        If True, return the mean single-channel waveform using the best channel
        for `cluster_id`. Otherwise, return the multi-channel waveform with
        all data channels included.

    Returns
    -------
    mean_wave : np.ndarray
        Mean waveform with shape (nt,) if `best = True`, or shape (n_chans,nt)
        otherwise.
    spike_subset : np.ndarray
        Indices for randomly selected spikes, relative to the list of spike
        times assigned to this cluster. In other words, `spike_subset = [0,1,2]`
        would indicate that the first, second and third spikes from this cluster
        were used (in ascending order of spike time).
    
    """
    results_dir = Path(results_dir)
    if best:
        chan = get_best_channels(results_dir)[cluster_id]
    else:
        chan = None

    spike_times, spike_subset = get_cluster_spikes(cluster_id, results_dir,
                                                   n_spikes=n_spikes)
    waves = get_spike_waveforms(spike_times, results_dir, bfile=bfile, chan=chan)
    mean_wave = waves.mean(axis=-1)

    return mean_wave, spike_subset


def get_best_channels(results_dir):
    """Get channel numbers with largest template norm for each cluster."""
    templates = np.load(results_dir / 'templates.npy')
    best_chans = (templates**2).sum(axis=1).argmax(axis=-1)
    return best_chans

def get_best_channel(results_dir, cluster_id):
    return get_best_channels(results_dir)[cluster_id]

def get_cluster_spikes(cluster_id, results_dir, n_spikes=np.inf):
    """Get `n_spikes` random spike times assigned to `cluster_id`."""
    spike_times = np.load(results_dir / 'spike_times.npy')
    spike_clusters = np.load(results_dir / 'spike_clusters.npy')
    spike_idx = (spike_clusters == cluster_id).nonzero()[0]
    if n_spikes != np.inf:
        spike_subset = np.random.choice(
            np.arange(spike_idx.size), min(spike_idx.size, n_spikes), replace=False
            )
    else:
        spike_subset = np.arange(spike_idx.size)
    spike_idx = spike_idx[spike_subset]
    spike_idx.sort()
    spikes = spike_times[spike_idx]

    return spikes, spike_subset


def get_spike_waveforms(spikes, results_dir, bfile=None, chan=None):
    """Get waveform for each spike in `spikes`, multi- or single-channel.
    
    Parameters
    ----------
    spikes : list or array-like
        Spike times (in units of samples) for the desired waveforms, from
        `spike_times.npy`.
    results_dir : str or Path
        Path to directory where Kilosort4 sorting results were saved.
    bfile : kilosort.io.BinaryFiltered; optional
        Kilosort4 data file object. By default, this will be loaded using the
        information in `ops.npy` in the saved results.
    chan : int; optional.
        Channel to use for single-channel waveforms. If not specified, all
        channels will be returned.

    Returns
    -------
    waves : np.ndarray
        Array of spike waveforms with shape `(nt, len(spikes))`.
    
    """
    if isinstance(spikes, int):
        spikes = [spikes]

    if bfile is None:
        ops = io.load_ops(results_dir / 'ops.npy')
        bfile = io.bfile_from_ops(ops)
    whitening_mat_inv = np.load(results_dir / 'whitening_mat_inv.npy')

    waves = []
    for t in spikes:
        tmin = max(t - bfile.nt0min, 0)
        tmax = t + (bfile.nt - bfile.nt0min)
        w = bfile[tmin:tmax].cpu().numpy()
        if whitening_mat_inv is not None:
            w = whitening_mat_inv @ w
        if w.shape[1] == bfile.nt:
            # Don't include spikes at the start or end of the recording that
            # get truncated to fewer time points.
            waves.append(w)
    waves = np.stack(waves, axis=-1)

    if chan is not None:
        waves = waves[chan,:]

    return waves


def cluster_templates(cluster_id, results_dir, mean=False, best=False, spike_subset=None):
    """Get template-like centroid for this `cluster_id`, scaled for each spike.
    
    Note that template here actually refers to the final clusters. The
    template-like structure is obtained from the multi-channel waveforms of all
    spikes assigned to the cluster. The 'template' is scaled separately for each
    spike based on its amplitude.

    Parameters
    ----------
    cluster_id : int
        Cluster index to reference from `spike_clusters.npy` in the results
        directory.
    results_dir : str or Path
        Path to directory where Kilosort4 sorting results were saved.
    mean : bool; default=False
        If True, return the mean 'template' across spikes.
    best : bool; default=False
        If True, return single channel 'template(s)' for this cluster_id using
        the channel with the largest norm.
    spike_subset : np.ndarray; optional.
        Array of indices specifying which spikes to use, as returned by
        `get_cluster_spikes` or `mean_waveform`. This helps reduce processing
        time for large datasets if only a subset of the spikes are
        ultimately used.
        
        For example, `spike_subset = [0,2,5]` would use the first, third,
        and sixth spike (in order of increasing spike time), regardless of
        what the actual spike times are.

    Return
    ------
    temps : np.ndarray
        Array of templates with shape `(n_spikes, nt, n_channels)`, or
        with shape `(nt, n_channels)` if `mean=True`. If `best=True`,
        then `n_channels=1`.
        
    """
    results_dir = Path(results_dir)
    spike_clusters = np.load(results_dir / 'spike_clusters.npy')
    spike_idx = (spike_clusters == cluster_id).nonzero()[0]
    if spike_subset is not None:
        spike_idx = spike_idx[spike_subset]
    temps = get_templates(spike_idx, results_dir)
    if best:
        chan = get_best_channel(results_dir, cluster_id)
        temps = temps[:,:,chan]
    if mean:
        return temps.mean(axis=0)

    return temps


def get_templates(spike_idx, results_dir):
    """Get template-like centroids for clusters assigned to one or more spikes.
    
    Parameters
    ----------
    spike_idx : int or array-like
        Index or list/array of indices into `spike_times.npy`
    results_dir : str or Path
        Path to directory where Kilosort4 sorting results were saved.
    
    Returns
    -------
    scaled : np.ndarray
        Array of templates with shape `(len(spike_idx), nt, n_channels)`.
        Templates are scaled using `amplitude.npy` to match the scale of
        unwhitened spike waveforms.

    """
    results_dir = Path(results_dir)
    if isinstance(spike_idx, int):
        spike_idx = [spike_idx]
    templates = np.load(results_dir / 'templates.npy')
    # Note that spike_clusters.npy is identical to spike_templates.npy for KS4
    spike_templates = np.load(results_dir / 'spike_clusters.npy')
    amplitudes = np.load(results_dir / 'amplitudes.npy')
    template_idx = spike_templates[spike_idx]
    temps = templates[template_idx, :, :]
    scaled = amplitudes[spike_idx, np.newaxis, np.newaxis] * temps

    return scaled


def get_good_cluster(results_dir, n=1):
    """Pick `n` random cluster ids with a label of 'good.'"""
    labels = get_labels(results_dir)
    labels = [x for x in labels if x[1] == 'good']
    rows = np.random.choice(
        np.arange(len(labels)), size=min(n, len(labels)), replace=False
        )
    cluster_ids = [int(labels[r][0]) for r in rows]
    if n == 1:
        cluster_ids = cluster_ids[0]

    return cluster_ids


def get_labels(results_dir):
    """Load good/mua labels as a list of ['cluster', 'label'] pairs."""
    results_dir = Path(results_dir)
    filename = results_dir / 'cluster_KSLabel.tsv'
    with open(filename) as f:
        text = f.read()
    rows = text.split('\n')
    labels = [r.split('\t') for r in rows[1:]][:-1]

    return labels


def get_refractory(results_dir):
    """Get list of refractory (good) units and their estimated contamination."""

    clu = np.load(results_dir / 'spike_clusters.npy')
    st = np.load(results_dir / 'spike_times.npy')
    ops = io.load_ops(results_dir / 'ops.npy')

    is_ref, est_contam_rate = CCG.refract(
        clu, st/ops['fs'],
        acg_threshold=ops['acg_threshold'],
        ccg_threshold=ops['ccg_threshold']
        )
    
    return is_ref.astype(bool), est_contam_rate
