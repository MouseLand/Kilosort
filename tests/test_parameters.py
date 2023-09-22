'''Check that configurable parameters are propagated correctly.'''

import numpy as np
from kilosort.spikedetect import template_centers


def test_dmin():
    settings = {'dmin': None, 'dminx': None}
    ops = {'xc': np.array([1, 2, 3]), 'yc': np.array([4, 5, 6]),
           'settings': settings}
    ops = template_centers(ops)
    assert ops['dmin'] is not None  # set based on xc, yc
    assert ops['dminx'] is not None
    assert ops['settings']['dmin'] is None  # shouldn't change
    assert ops['settings']['dminx'] is None

    settings = {'dmin': 5, 'dminx': 7}
    ops = {'xc': np.array([1, 2, 3]), 'yc': np.array([4, 5, 6]),
           'settings': settings}
    ops = template_centers(ops)
    assert ops['dmin'] == 5
    assert ops['dminx'] == 7
