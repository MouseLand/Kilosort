from numba import njit
from numba.types import bool_
import numpy as np
import torch


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
