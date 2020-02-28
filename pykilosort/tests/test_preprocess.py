from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import lfilter as lfilter_cpu, convolve as convolve_cpu
import cupy as cp

from ..cptools import median, lfilter, svdecon, svdecon_cpu, convolve, free_gpu_memory
from ..preprocess import my_conv2


sig = 250
tmax = 1000
test_path = Path('/home/olivier/Documents/PYTHON/pykilosort/pykilosort/tests')


def test_another_convolve():

    s1 = np.load(test_path.joinpath('my_conv2_input.npy'))
    s1_expected = np.load(test_path.joinpath('my_conv2_output.npy'))
    s1_matlab = np.load(test_path.joinpath('my_conv2_output_matlab.npy'))
    # out = my_conv2(cp.asarray(s1)[:, np.newaxis], 250, 0)

    #
    # plt.plot(cp.asnumpy(out))
    # plt.plot(s1_expected[:, :2])
    # plt.plot(s1_matlab[:, :2])

def create_test_dataset():
    cp = np  # cpu mode only here
    s1 = np.load(test_path.joinpath('my_conv2_input.npy'))
    s0 = np.copy(s1)
    
    tmax = np.ceil(4 * sig)
    
    dt = cp.arange(-tmax, tmax + 1)
    gauss = cp.exp(-dt ** 2 / (2 * sig ** 2))
    gauss = gauss / cp.sum(gauss)
    cNorm = lfilter_cpu(gauss, 1, np.r_[np.ones(s1.shape[0]), np.zeros(int(tmax))])

    cNorm = cNorm[int(tmax):]
    # plt.plot(cNorm), plt.show()

    s1 = lfilter_cpu(gauss, 1, np.r_[s1, np.zeros((int(tmax), s1.shape[1]))], axis=0)
    s1 = s1[int(tmax):] / cNorm[:, np.newaxis]

    # import matplotlib.pyplot as plt
    # plt.plot(s0)
    # plt.plot(s1)
    np.save(test_path.joinpath('my_conv2_input.npy'), s0)
    np.save(test_path.joinpath('my_conv2_output.npy'), s1)
