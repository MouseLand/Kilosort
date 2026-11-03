"""Checks the PC features exported for Phy by `make_pc_features`."""

import numpy as np
import torch

from kilosort.postprocessing import make_pc_features


def _toy_ops(n_chan=16, n_near=4, template_chans=(0, 2, 4, 6, 8, 10)):
    # Channel c's neighbourhood is channels c..c+3; template j sits on
    # channel template_chans[j].
    iCC = (torch.arange(n_near)[:, None] + torch.arange(n_chan)[None, :]) % n_chan
    return {
        'iCC': iCC,
        'iU': torch.tensor(template_chans),
        'xc': np.zeros(n_chan, dtype='float32'),
        'yc': (20 * np.arange(n_chan)).astype('float32'),
        'dmin': 20, 'dminx': 32, 'nearest_chans': n_near,
        }


def _expected(tF, ops, spike_templates, spike_clusters, feature_ind):
    """What Phy should see: each spike's own features on the channels listed
    for its cluster, and zeros on channels its template did not cover."""
    iC = ops['iCC'][:, ops['iU']].numpy()
    n_spikes, _, n_pcs = tF.shape
    out = np.zeros((n_spikes, n_pcs, feature_ind.shape[1]), tF.numpy().dtype)
    for s in range(n_spikes):
        t, c = spike_templates[s], spike_clusters[s]
        for k, ch in enumerate(feature_ind[c]):
            pos = np.flatnonzero(iC[:, t] == ch)
            if pos.size:
                out[s, :, k] = tF[s, pos[0]].numpy()
    return out


class TestMakePcFeatures:
    def test_templates_shared_between_clusters(self):
        # Template 1 has spikes in clusters 0 and 1, and template 3 in
        # clusters 1 and 2, as happens when final clustering splits a
        # template's spikes. Each spike must keep its own features, laid out
        # by its own cluster's channels.
        ops = _toy_ops()
        rng = np.random.default_rng(0)
        templates_of = {0: [0, 1], 1: [1, 2, 3], 2: [3, 4, 5]}
        spike_clusters = np.repeat([0, 1, 2], 200).astype('int32')
        spike_templates = np.concatenate(
            [rng.choice(templates_of[c], 200) for c in range(3)]
            ).astype('int32')
        perm = rng.permutation(spike_clusters.size)   # interleave in time
        spike_clusters, spike_templates = spike_clusters[perm], spike_templates[perm]
        tF = torch.from_numpy(rng.normal(size=(600, 4, 3)).astype('float32'))
        tF[:, 0] += 5   # most energy on each template's own channel

        pc, ind = make_pc_features(ops, spike_templates, spike_clusters, tF.clone())

        assert pc.shape == (600, 3, 4)
        expected = _expected(tF, ops, spike_templates, spike_clusters, ind)
        assert np.array_equal(pc.numpy(), expected)
        # Each cluster's main channel (that of its first template) is listed.
        iU = ops['iU'].numpy()
        for c, ts in templates_of.items():
            assert iU[ts[0]] in ind[c]
