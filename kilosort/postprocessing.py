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
    tmass = (tF**2).sum(-1)
    tmass = tmass / tmass.sum(1, keepdim=True)
    xc = torch.from_numpy(ops['xc']).to(tmass.device)
    yc = torch.from_numpy(ops['yc']).to(tmass.device)
    chs = ops['iCC'][:, ops['iU'][st[:,1]]].cpu()
    xc0 = xc[chs.T]
    yc0 = yc[chs.T]

    xs = (xc0 * tmass).sum(1).cpu().numpy()
    ys = (yc0 * tmass).sum(1).cpu().numpy()

    return xs, ys


def make_pc_features(ops, spike_templates, spike_clusters, tF):
    # spike_templates: st[:,1]
    # spike clusters:  clu

    xy, iC = xy_templates(ops)
    n_clusters = np.unique(spike_clusters).size
    feature_ind = np.zeros((n_clusters, 10), dtype=np.uint32)

    for i in np.unique(spike_clusters):
        iunq = np.unique(spike_templates[spike_clusters==i]).astype(int)
        ix = torch.from_numpy(np.zeros(int(spike_templates.max())+1, bool))
        ix[iunq] = True
        Xd, ch_min, ch_max, igood = get_data_cpu(
            ops, xy, iC, spike_templates, tF, None, None,
            dmin=ops['dmin'], dminx=ops['dminx'], ix=ix, merge_dim=False
            )

        # Take mean of Xd across spikes, find channels w/ largest norm
        spike_mean = Xd.mean(0)
        chan_norm = torch.linalg.norm(spike_mean, dim=1)
        sorted_chans, ind = torch.sort(chan_norm, descending=True)
        # Assign Xd to overwrite tF in-place
        tF[igood,:] = Xd[:, ind[:10], :]
        # Save channel inds for phy
        feature_ind[i,:] = ind[:10].numpy() + ch_min.cpu().numpy()
        # TODO: should be sorted by physical distance from first channel?
        # TODO: cast to uint32

    tF = torch.permute(tF, (0, 2, 1))

    return tF, feature_ind
