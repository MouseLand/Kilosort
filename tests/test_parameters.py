'''Check that configurable parameters are propagated correctly.'''

import numpy as np

from kilosort.spikedetect import template_centers
from kilosort.io import load_probe
from kilosort.utils import PROBE_DIR


def test_dmin():
    settings = {'dmin': None, 'dminx': 32}
    ops = {'xc': np.array([10, 20, 30]), 'yc': np.array([40, 40, 60]),
           'settings': settings, 'kcoords': np.array([0, 0, 0])}
    ops = template_centers(ops)
    assert ops['dmin'] is not None  # set based on xc, yc
    assert ops['dminx'] is not None
    assert ops['settings']['dmin'] is None  # shouldn't change
    
    # Neuropixels 1 3B1 (4 columns with stagger)
    np1_probe = load_probe(PROBE_DIR / 'neuropixPhase3B1_kilosortChanMap.mat')
    ops = {'xc': np1_probe['xc'], 'yc': np1_probe['yc'], 
           'kcoords': np1_probe['kcoords'], 'settings': settings}
    ops = template_centers(ops)
    assert ops['dmin'] == 20     # Median vertical spacing of contacts
    assert ops['xup'].size == 4  # Number of lateral pos for universal templates

    # Just one shank of NP2
    np2_probe = load_probe(PROBE_DIR / 'NP2_kilosortChanMap.mat')
    ops = {'xc': np2_probe['xc'], 'yc': np2_probe['yc'],
           'kcoords': np2_probe['kcoords'], 'settings': settings}
    ops = template_centers(ops)
    assert ops['dmin'] == 15
    assert ops['xup'].size == 3

    # Linear probe
    lin_probe = load_probe(PROBE_DIR / 'Linear16x1_kilosortChanMap.mat')
    ops = {'xc': lin_probe['xc'], 'yc': lin_probe['yc']*20,
           'kcoords': lin_probe['kcoords'], 'settings': settings}
    ops = template_centers(ops)
    assert ops['dmin'] == 20
    assert ops['xup'].size == 1

    settings = {'dmin': 5, 'dminx': 7}
    ops = {'xc': np.array([10, 20, 30]), 'yc': np.array([40, 40, 60]),
           'settings': settings, 'kcoords': np.array([0, 0, 0])}
    ops = template_centers(ops)
    assert ops['dmin'] == 5
    assert ops['dminx'] == 7
