import numpy as np
from ..preprocess import get_good_channels, get_whitening_matrix, preprocess
from ..utils import p, read_data


def test_good_channels(raw_data, probe, params):
    igood = get_good_channels(raw_data, probe, params)
    assert np.sum(~igood) == 81
    assert len(igood) == 374


def test_whitening_matrix(data_path, raw_data, probe_good, params):
    Wrot = get_whitening_matrix(raw_data, probe_good, params)
    Wrot_mat = np.load(data_path / 'whitening_matrix.npy')
    assert np.allclose(Wrot, Wrot_mat, atol=10)


def test_preprocess_1(data_path, raw_data, probe_good, params):
    # TODO: use probe and not probe_good
    preprocess(
        raw_data=raw_data, probe=probe_good, params=params, proc_path=data_path / 'proc.dat')


def test_preprocess_2(data_path):
    shape = (-1, 293)
    shape = None
    dat_py = read_data(data_path / 'proc.dat', shape=shape, dtype=np.int16, axis=0)
    dat_mat = read_data(data_path / 'temp_wh.dat', shape=shape, dtype=np.int16, axis=0)
    assert np.median(np.abs(dat_py[::100] - dat_mat[::100])) == 0.
