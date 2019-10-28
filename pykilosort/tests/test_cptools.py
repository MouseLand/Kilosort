import numpy as np
from scipy.signal import lfilter as lfilter_cpu
import cupy as cp

from ..cptools import median, lfilter
# from ..utils import p


def test_median_1(dtype, axis):
    arr = cp.random.rand(2000, 1000).astype(dtype)
    m1 = cp.asnumpy(median(arr, axis=axis))
    m2 = np.median(cp.asnumpy(arr), axis=axis)
    assert np.allclose(m1, m2)


def test_lfilter_1():
    b = (0.96907117, -2.90721352, 2.90721352, -0.96907117)
    a = (1., -2.93717073, 2.87629972, -0.93909894)
    arr = np.random.rand(1000, 100).astype(np.float32)

    fil_gpu = cp.asnumpy(lfilter(b, a, cp.asarray(arr), axis=0))
    fil_cpu = lfilter_cpu(b, a, arr, axis=0)

    assert np.allclose(fil_cpu, fil_gpu, atol=.2)
