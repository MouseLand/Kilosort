from pathlib import Path
import numpy as np
from scipy.signal import lfilter as lfilter_cpu
import cupy as cp

from ..preprocess import my_conv2


sig = 250
tmax = 1000
test_path = Path('/home/olivier/Documents/PYTHON/pykilosort/pykilosort/tests')


def test_convolve():

    s1 = np.load(test_path.joinpath('my_conv2_input.npy'))
    s1_expected = np.load(test_path.joinpath('my_conv2_output.npy'))

    def assert_sim(out):
        assert np.all((cp.asnumpy(out[1000:-1000, :1]) - s1_expected[1000:-1000, :1]) < 1e-6)

    x = cp.asarray(s1)
    assert_sim(my_conv2(x, 250, 0, nwin=0, pad=None))
    assert_sim(my_conv2(x, 250, 0, nwin=10000, pad='zeros'))
    assert_sim(my_conv2(x, 250, 0, nwin=10000, pad='constant'))
    assert_sim(my_conv2(x, 250, 0, nwin=10000, pad='flip'))
    # TODO add a random array as well (seed)
    # TODO make tapers for both full and swapped version
    # TODO try VPN with Cyrille
    # TODO how to compare spike sorters
    # import matplotlib.pyplot as plt
    # plt.plot(cp.asnumpy(out[:, :1]))
    # plt.plot(cp.asnumpy(out[:, :1]))
    # plt.plot(s1_expected[:, :1])
    # diff = cp.asnumpy(out[:, :1]) - s1_expected[:, :1]
    # plt.plot(diff)


def create_test_dataset():
    cp = np  # cpu mode only here
    s1 = np.load(test_path.joinpath('my_conv2_input.npy'))
    s0 = np.copy(s1)
    tmax = np.ceil(4 * sig)
    dt = cp.arange(-tmax, tmax + 1)
    gauss = cp.exp(-dt ** 2 / (2 * sig ** 2))
    gauss = (gauss / cp.sum(gauss)).astype(np.float32)

    cNorm = lfilter_cpu(gauss, 1., np.r_[np.ones(s1.shape[0]), np.zeros(int(tmax))])
    cNorm = cNorm[int(tmax):]

    s1 = lfilter_cpu(gauss, 1, np.r_[s1, np.zeros((int(tmax), s1.shape[1]))], axis=0)
    s1 = s1[int(tmax):] / cNorm[:, np.newaxis]

    # import matplotlib.pyplot as plt
    # plt.plot(s0)
    # plt.plot(s1)
    np.save(test_path.joinpath('my_conv2_input.npy'), s0)
    np.save(test_path.joinpath('my_conv2_output.npy'), s1)
