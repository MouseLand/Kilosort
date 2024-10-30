from pathlib import Path
import numpy as np
from kilosort import io


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
    bfile : kilosort.io.BinaryFiltered; optional.
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
    
    """
    results_dir = Path(results_dir)

    ops = io.load_ops(results_dir / 'ops.npy')
    whitening_mat_inv = np.load(results_dir / 'whitening_mat_inv.npy')
    spike_times = np.load(results_dir / 'spike_times.npy')
    spike_clusters = np.load(results_dir / 'spike_clusters.npy')

    if best:
        templates = np.load(results_dir / 'templates.npy')
        chan = (templates**2).sum(axis=1).argmax(axis=-1)[cluster_id]
    else:
        chan = None
    if bfile is None:
        bfile = io.bfile_from_ops(ops)

    spikes = get_cluster_spikes(
        cluster_id, spike_times, spike_clusters, n_spikes=n_spikes
        )
    waves = get_spike_waveforms(
        spikes, bfile, whitening_mat_inv=whitening_mat_inv, chan=chan
        )
    mean_wave = waves.mean(axis=-1)

    return mean_wave


def get_cluster_spikes(cluster_id, spike_times, spike_clusters, n_spikes=np.inf):
    """Get `n_spikes` random spike times assigned to `cluster_id`."""
    spikes = spike_times[spike_clusters == cluster_id]
    spikes = np.random.choice(spikes, min(spikes.size, n_spikes), replace=False)
    return spikes


def get_spike_waveforms(spikes, bfile, whitening_mat_inv=None, chan=None):
    """Get waveform for each spike in `spikes`, multi- or single-channel."""
    waves = []
    for t in spikes:
        tmin = t - bfile.nt0min
        tmax = t + (bfile.nt - bfile.nt0min) + 1
        w = bfile[tmin:tmax].cpu().numpy()
        if whitening_mat_inv is not None:
            w = whitening_mat_inv @ w
        waves.append(w)
    waves = np.stack(waves, axis=-1)

    if chan is not None:
        waves = waves[chan,:]

    return waves
