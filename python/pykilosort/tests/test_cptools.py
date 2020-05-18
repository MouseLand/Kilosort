import numpy as np
from numpy.testing import assert_allclose as ac
from scipy.signal import lfilter as lfilter_cpu
import cupy as cp

from pytest import fixture

from ..cptools import (
    median, lfilter, svdecon, svdecon_cpu, free_gpu_memory,
    convolve_cpu, convolve_gpu, convolve_gpu_direct, convolve_gpu_chunked)


def test_median_1(dtype, axis):
    arr = cp.random.rand(2000, 1000).astype(dtype)
    m1 = cp.asnumpy(median(arr, axis=axis))
    m2 = np.median(cp.asnumpy(arr), axis=axis)
    assert np.allclose(m1, m2)


def test_lfilter_1():
    tmax = 1000
    dt = np.arange(-tmax, tmax + 1)
    gaus = np.exp(-dt ** 2 / (2 * 250 ** 2))
    b = gaus / np.sum(gaus)

    a = 1.

    n = 2000
    arr = np.r_[np.ones(n), np.zeros(n)]

    fil_gpu = cp.asnumpy(lfilter(b, a, cp.asarray(arr), axis=0)).ravel()
    fil_cpu = lfilter_cpu(b, a, arr, axis=0)

    assert np.allclose(fil_cpu, fil_gpu, atol=1e-6)


def test_lfilter_2():
    b = (0.96907117, -2.90721352, 2.90721352, -0.96907117)
    a = (1., -2.93717073, 2.87629972, -0.93909894)
    arr = np.random.rand(1000, 100).astype(np.float32)

    fil_gpu = cp.asnumpy(lfilter(b, a, cp.asarray(arr), axis=0))
    fil_cpu = lfilter_cpu(b, a, arr, axis=0)

    assert np.allclose(fil_cpu, fil_gpu, atol=.2)


def test_svdecon_1():
    X = cp.random.rand(10, 10)

    U, S, V = svdecon(X)
    Un, Sn, Vn = svdecon_cpu(X)

    assert np.allclose(S, Sn)


@fixture(params=(100, 2_000, 10_000, 50_000, 250_000))
def arr(request):
    return np.random.randn(int(request.param), 100)


@fixture
def gaus():
    dt = np.arange(-1000, 1000 + 1)
    gaus = np.exp(-dt ** 2 / (2 * 250 ** 2))
    return gaus / np.sum(gaus)


@fixture(params=(None, 'zeros', 'flip', 'constant'))
def pad(request):
    return request.param


@fixture
def nwin():
    return 2500


def test_convolve(arr, gaus, pad, nwin):
    free_gpu_memory()
    npad = gaus.shape[0] // 2

    # Upload the arrays to the GPU.
    arr_gpu = cp.asarray(arr)
    gaus_gpu = cp.asarray(gaus)

    # Compute the convolution on the CPU.
    conv_cpu = convolve_cpu(arr, gaus)

    def check(y):
        ac(cp.asnumpy(y)[npad:-npad, :], conv_cpu[npad:-npad, :], atol=1e-3)

    # Check the GPU direct version with the CPU version.
    check(convolve_gpu_direct(arr_gpu, gaus_gpu, pad=pad))

    if arr.shape[0] > nwin:
        # Check the GPU chunked version with the CPU version.
        y = convolve_gpu_chunked(arr_gpu, gaus_gpu, pad=pad, nwin=nwin)
        check(y)

        # DEBUG
        # import matplotlib.pyplot as plt
        # plt.plot(cp.asnumpy(conv_cpu[npad:-npad, 0]))
        # plt.plot(cp.asnumpy(y[npad:-npad, 0]))
        # plt.show()

    if pad is None:
        # This function should automatically route to the correct GPU implementation.
        check(convolve_gpu(arr_gpu, gaus_gpu))
