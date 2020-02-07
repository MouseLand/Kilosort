import numpy as np
from scipy.signal import lfilter as lfilter_cpu, convolve as convolve_cpu
import cupy as cp

from ..cptools import median, lfilter, svdecon, svdecon_cpu, convolve, free_gpu_memory


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


def test_convolve():
    for n in (1000, 10000, 100000, 250000, 450000):
        free_gpu_memory()
        x = cp.random.randn(n, 96)
        b = cp.sin(-cp.linspace(0.0, 5.0, 100))
        y = cp.asnumpy(convolve(x, b))

        for k in (0, 50, 95):
            y_cpu = convolve_cpu(cp.asnumpy(x)[:, k], cp.asnumpy(b), mode='same')
            assert np.max(np.abs(y[:, k] - y_cpu)) < 1e-8
