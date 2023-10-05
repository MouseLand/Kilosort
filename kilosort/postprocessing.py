from numba import njit
from numba.types import bool_
import numpy as np


@njit
def remove_duplicates(spike_times, spike_clusters, dt=15):
    '''Removes same-cluster spikes that occur within `dt` samples.'''
    keep = np.zeros_like(spike_times, bool_)
    cluster_t0 = {}
    for (i,t), c in zip(enumerate(spike_times), spike_clusters):
        t0 = cluster_t0.get(c, t-dt)
        if t >= (t0 + dt):
            # Separate spike, reset t0 and keep spike
            cluster_t0[c] = t
            keep[i] = True
        else:
            # Same spike, toss it out
            continue
    
    return spike_times[keep], spike_clusters[keep], keep
