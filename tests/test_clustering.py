import numpy as np
import pytest
import torch

from kilosort.clustering_qr import assign_iclust, x_centers
from kilosort.io import load_probe
from kilosort.utils import PROBE_DIR


def _synth_assign_inputs(n_spikes, nsub, nclust, n_neigh, seed, device):
    """Build a valid, self-consistent input set for ``assign_iclust``.

    Shapes mirror what ``clustering_qr.cluster`` passes: ``rows_neigh`` and
    ``tones2`` are (n_spikes, n_neigh); ``kn`` indexes the (nsub) graph nodes;
    ``isub`` is the per-node cluster label; ``ki``/``kj`` are the per-spike and
    per-node degrees (len(kj) must equal nsub == len(isub)).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    rows_neigh = torch.arange(n_spikes, device=device).unsqueeze(-1).tile((1, n_neigh))
    tones2 = torch.ones((n_spikes, n_neigh), device=device)
    kn = torch.randint(0, nsub, (n_spikes, n_neigh), generator=g, device=device)
    isub = torch.randint(0, nclust, (nsub,), generator=g, device=device)
    ki = torch.rand(n_spikes, generator=g, device=device) + 0.5
    kj = torch.rand(nsub, generator=g, device=device) + 0.5
    m = float(n_spikes * n_neigh)
    return rows_neigh, isub, kn, tones2, ki, kj, m


class TestAssignIclustChunking:
    """The chunked ``assign_iclust`` path must be identical to the unchunked path."""

    @pytest.mark.parametrize("lam", [0, 1])
    @pytest.mark.parametrize("chunk", [1, 7, 333, 999, 1000, 1024, 5000, 5001])
    def test_chunked_matches_unchunked(self, lam, chunk):
        device = torch.device("cpu")
        n_spikes, nsub, nclust, n_neigh = 5000, 800, 60, 10
        rows_neigh, isub, kn, tones2, ki, kj, m = _synth_assign_inputs(
            n_spikes, nsub, nclust, n_neigh, seed=0, device=device
        )

        full = assign_iclust(
            rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj,
            device=device, chunk=None,
        )
        chunked = assign_iclust(
            rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj,
            device=device, chunk=chunk,
        )

        assert chunked.shape == full.shape
        assert chunked.dtype == full.dtype
        # chunk >= n_spikes falls through to the fast path; smaller chunks exercise
        # the row loop (including a ragged final chunk for non-divisor sizes).
        assert torch.equal(full, chunked)


def random_np2(n_chans=384, n_shanks=4):
    # Generates xc,yc for a probe containing *all* neuropixels 2 contact positions,
    # then randomly subsamples from those positions to get a probe layout
    # corresponding to 384-channel output data.

    # 12um square contacts with 32um lateral spacing,
    # 15um vertical spacing,
    # 1280 contacts per shank

    # Want alternating 6um, 38um for lateral positions
    xc0 = np.empty(1280)
    xc0[::2] = 6
    xc0[1::2] = 38
    # Then add 250um for each additional shank
    xc = np.concatenate([xc0 + (250*i) for i in range(4)])

    # For vertical positions, start at 6 and increase by 15
    yc0 = (np.arange(640)*15) + 6
    # Each position appears twice (two columns on each shank)
    yc0 = np.repeat(yc0, 2)
    yc = np.concatenate([yc0 for i in range(4)])

    # Repeat 0 1280 times, then repeat 1 1280 times, etc
    kcoords = np.repeat(np.arange(4), 1280)

    # Pick n_chans out of n_shanks
    shanks_used = np.random.choice(range(4), n_shanks, replace=False)
    shank_indices = np.argwhere(np.isin(kcoords, shanks_used))[:,0]
    contact_indices = np.random.choice(shank_indices, n_chans, replace=False)

    return {'xc': xc[contact_indices], 'yc': yc[contact_indices]}


class TestCenters:
    ops = {'dminx': 32}

    def test_linear(self, data_directory):
        # NOTE: The `data_directory` argument is only there to make sure probes are
        # downloaded before these tests are run.
        probe = load_probe(PROBE_DIR/'Linear16x1_test.mat')
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        # X positions are all 1um
        assert len(centers) == 1
        assert np.abs(centers[0] - 1) < 5

    def test_np1(self):
        probe = load_probe(PROBE_DIR/'NeuroPix1_default.mat')
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        # One shank from 11um to 59um, should be 1 center near 35um
        assert len(centers) == 1
        assert np.abs(centers[0] - 35) < 5

    def test_np2_1shank(self):
        probe = load_probe(PROBE_DIR/'NeuroPix2_default.mat')
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        # One shank from 0 to 32um, should be 1 center near 16um
        assert len(centers) == 1
        assert np.abs(centers[0] - 16) < 5

    def test_np2_3shank(self):
        probe = random_np2(n_shanks=3)
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        assert len(centers == 3)
        true = np.array([22, 272, 522, 772])
        for c in centers:
            # Each center is within 2 microns of exactly one true center
            print(f'center: {c}')
            assert (np.abs(c - true) < 5).sum() == 1

    def test_np2_4shank(self):
        probe = random_np2(n_shanks=4)
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        # All centers should be within 2 microns of the true values
        print(f'centers: {centers}')
        assert np.allclose(np.sort(centers), np.sort([22, 272, 522, 772]), atol=5)
