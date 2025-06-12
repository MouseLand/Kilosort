from numba import njit
from numba.types import bool_
import numpy as np
import torch

from kilosort.clustering_qr import xy_templates, get_data_cpu


@njit("(int64[:], int32[:], int32)")
def remove_duplicates(spike_times, spike_clusters, dt=15):
    '''Removes same-cluster spikes that occur within `dt` samples.'''
    keep = np.zeros_like(spike_times, bool_)
    cluster_t0 = {}
    for i in range(spike_times.size):
        t = spike_times[i]
        c = spike_clusters[i]
        if c in cluster_t0:
            t0 = cluster_t0[c]
        else:
            t0 = t - dt

        if t >= (t0 + dt):
            # Separate spike, reset t0 and keep spike
            cluster_t0[c] = t
            keep[i] = True
        else:
            # Same spike, toss it out
            continue
    
    return spike_times[keep], spike_clusters[keep], keep


def compute_spike_positions(st, tF, ops):
    '''Get x,y positions of spikes relative to probe.'''
    # Determine channel weightings for nearest channels
    # based on norm of PC features. Channels that are far away have 0 weight,
    # determined by `ops['settings']['position_limit']`.

    # Get indexing variables from ops and move to CPU (same as tF).
    cpu = torch.device('cpu')
    icc_mask = ops['iCC_mask'].to(cpu)
    iU = ops['iU'].to(cpu)
    iCC = ops['iCC'].to(cpu)

    # Weightings from norm of temporal features
    tmass = torch.norm(tF, 2, dim=-1)
    tmask = icc_mask[:, iU[st[:,1]]].T
    tmass = tmass * tmask
    tmass = tmass / tmass.sum(1, keepdim=True)

    # Get x,y coordinates of nearest channels.
    xc = torch.from_numpy(ops['xc'])
    yc = torch.from_numpy(ops['yc'])
    chs = iCC[:, iU[st[:,1]]]
    xc0 = xc[chs.T]
    yc0 = yc[chs.T]

    # Estimate spike positions as weighted sum of coordinates of nearby channels.
    xs = (xc0 * tmass).sum(1).numpy()
    ys = (yc0 * tmass).sum(1).numpy()

    return xs, ys


def make_pc_features(ops, spike_templates, spike_clusters, tF):
    '''Get PC Features and corresponding indices for export to Phy.

    NOTE: This function will update tF in-place!

    Parameters
    ----------
    ops : dict
        Dictionary of state variables updated throughout the sorting process.
        This function is intended to be used with the final state of ops, after
        all sorting has finished.
    spike_templates : np.ndarray
        Vector of template ids with shape `(n_spikes,)`. This is equivalent to
        `st[:,1]`, where `st` is returned by `template_matching.extract`.
    spike_clusters : np.ndarray
        Vector of cluster ids with shape `(n_pikes,)`. This is equivalent to
        `clu` returned by `template_matching.merging_function`.
    tF : torch.Tensor
        Tensor of pc features as returned by `template_matching.extract`,
        with shape `(n_spikes, nearest_chans, n_pcs)`.

    Returns
    -------
    tF : torch.Tensor
        As above, but with some data replaced so that features are associated 
        with the final clusters instead of templates. The second and third
        dimensions are also swapped to conform to the shape expected by Phy.
    feature_ind : np.ndarray
        Channel indices associated with the data present in tF for each cluster,
        with shape `(n_clusters, nearest_chans)`.
    
    '''

    # xy: template centers, iC: channels associated with each template
    xy, iC = xy_templates(ops)
    n_templates = iC.shape[1]
    n_clusters = np.unique(spike_clusters).size
    n_chans = ops['nearest_chans']
    feature_ind = np.zeros((n_clusters, n_chans), dtype=np.uint32)

    for i in np.unique(spike_clusters):
        # Get templates associated with cluster (often just 1)
        iunq = np.unique(spike_templates[spike_clusters==i]).astype(int)
        # Get boolean mask with size (n_templates,), True if they match cluster
        ix = torch.from_numpy(np.zeros(n_templates, bool))
        ix[iunq] = True
        # Get PC features for all spikes detected with those templates (Xd),
        # and the indices in tF where those spikes occur (igood).
        Xd, igood, ichan = get_data_cpu(
            ops, xy, iC, spike_templates, tF, None, None,
            dmin=ops['dmin'], dminx=ops['dminx'], ix=ix, merge_dim=False
            )

        # Take mean of features across spikes, find channels w/ largest norm
        spike_mean = Xd.mean(0)
        chan_norm = torch.linalg.norm(spike_mean, dim=1)
        sorted_chans, ind = torch.sort(chan_norm, descending=True)
        # Assign features to overwrite tF in-place
        tF[igood,:] = Xd[:, ind[:n_chans], :]
        # Save channel inds for phy
        feature_ind[i,:] = ichan[ind[:n_chans]].cpu().numpy()

    # Swap last 2 dimensions to get ordering Phy expects
    tF = torch.permute(tF, (0, 2, 1))

    return tF, feature_ind
