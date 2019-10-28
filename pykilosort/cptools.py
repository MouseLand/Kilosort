import ctypes
from math import ceil
from textwrap import dedent

import numpy as np
import cupy as cp


def make_kernel(kernel, name, **const_arrs):
    """Compile a kernel and pass optional constant ararys."""
    mod = cp.core.core.compile_with_cache(kernel, prepend_cupy_headers=False)
    b = cp.core.core.memory_module.BaseMemory()
    # Pass constant arrays.
    for n, arr in const_arrs.items():
        b.ptr = mod.get_global_var(n)
        p = cp.core.core.memory_module.MemoryPointer(b, 0)
        p.copy_from_host(arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes)
    return mod.get_function(name)


def get_lfilter_kernel(N, isfortran, reverse=False):
    order = 'f' if isfortran else 'c'
    return dedent("""
    const int N = %d;
    __constant__ float a[N + 1];
    __constant__ float b[N + 1];


    __device__ int get_idx_f(int n, int col, int n_samples, int n_channels) {
        return n_samples * col + n;  // Fortran order.
    }
    __device__ int get_idx_c(int n, int col, int n_samples, int n_channels) {
        return n * n_channels + col;  // C order.
    }

    // LTI IIR filter implemented using a difference equation.
    // see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    extern "C" __global__ void lfilter(
            const float* x, float* y, const int n_samples, const int n_channels){
        // Initialize the state variables.
        float d[N + 1];
        for (int k = 0; k <= N; k++) {
            d[k] = 0.0;
        }

        float xn = 0.0;
        float yn = 0.0;

        int idx = 0;

        // Column index.
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        for (int n = 0; n < n_samples; n++) {
            idx = get_idx_%s(%s, col, n_samples, n_channels);
            // Load the input element.
            xn = x[idx];
            // Compute the output element.
            yn = (b[0] * xn + d[0]) / a[0];
            // Update the state variables.
            for (int k = 0; k < N; k++) {
                d[k] = b[k + 1] * xn - a[k + 1] * yn + d[k + 1];
            }
            // Update the output array.
            y[idx] = yn;
        }
    }
    """ % (N, order, 'n' if not reverse else 'n_samples - 1 - n'))


def lfilter(b, a, arr, axis=0, reverse=False):
    """Perform a linear filter along the first axis on a GPU array."""

    assert isinstance(arr, cp.ndarray)
    assert axis == 0, "Only filtering along the first axis is currently supported."

    n_samples, n_channels = arr.shape

    block = (128,)
    grid = (int(ceil(n_channels / float(block[0]))),)

    b = np.array(b, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    kernel = get_lfilter_kernel(len(b) - 1, cp.isfortran(arr), reverse=reverse)

    lfilter = make_kernel(kernel, 'lfilter', b=b, a=a)

    y = cp.zeros_like(arr)
    lfilter(grid, block, (arr, y, y.shape[0], y.shape[1]))

    return y


def median(a, axis=0):
    """Compute the median of a CuPy array on the GPU."""
    a = cp.asarray(a)

    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]

    part = cp.partition(a, kth, axis=axis)

    if part.shape == ():
        # make 0-D arrays work
        return part.item()
    if axis is None:
        axis = 0

    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        # index with slice to allow mean (below) to work
        indexer[axis] = slice(index, index + 1)
    else:
        indexer[axis] = slice(index - 1, index + 1)

    return cp.mean(part[indexer], axis=axis)


def svdecon(X, nPC0=None):
    """
    Input:
    X : m x n matrix

    Output:
    X = U*S*V'

    Description:

    Does equivalent to svd(X,'econ') but faster

        Vipin Vijayan (2014)

    """

    m, n = X.shape

    nPC = nPC0 or min(m, n)

    if m <= n:
        C = cp.dot(X, X.T)
        D, U = cp.linalg.eigh(C, 'U')

        ix = cp.argsort(np.abs(D))[::-1]
        d = D[ix]
        U = U[:, ix]
        d = d[:nPC]
        U = U[:, :nPC]

        V = cp.dot(X.T, U)
        s = cp.sqrt(d)
        V = V / s.T
        S = cp.diag(s)
    else:
        C = cp.dot(X.T, X)
        D, V = cp.linalg.eigh(C)

        ix = cp.argsort(cp.abs(D))[::-1]
        d = D[ix]
        V = V[:, ix]

        # convert evecs from X'*X to X*X'. the evals are the same.
        U = cp.dot(X, V)
        s = cp.sqrt(d)
        U = U / s.T
        S = cp.diag(s)

    return U, S, V


def zscore(a, axis=0):
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=0)
    return (a - mns) / sstd


def free_gpu_memory():
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
