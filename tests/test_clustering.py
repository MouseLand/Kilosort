import numpy as np

from kilosort.clustering_qr import x_centers
from kilosort.io import load_probe
from kilosort.utils import PROBE_DIR


def random_np2(n_chans=384, n_shanks=4):
    # Generates a probe containing *all* neuropixels 2 contact positions,
    # then randomly subsamples from those positions to get a probe layout
    # corresponding to 384-channel output data.

    probe = {}
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
        probe = load_probe(PROBE_DIR/'Linear16x1_kilosortChanMap.mat')
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        # X positions are all 1um
        assert len(centers) == 1
        assert np.abs(centers[0] - 1) < 5

    def test_np1(self):
        probe = load_probe(PROBE_DIR/'neuropixPhase3B1_kilosortChanMap.mat')
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        # One shank from 11um to 59um, should be 1 center near 35um
        assert len(centers) == 1
        assert np.abs(centers[0] - 35) < 5

    def test_np2_1shank(self):
        probe = load_probe(PROBE_DIR/'NP2_kilosortChanMap.mat')
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
