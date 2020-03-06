from contextlib import redirect_stderr
import ctypes
import io
from functools import wraps
import logging
from math import ceil
from textwrap import dedent

import numpy as np
from scipy import signal as ss
import cupy as cp

logger = logging.getLogger(__name__)


# LTI filter on GPU (NOTE: inefficient and soon to be deprecated)
# ---------------------------------------------------------------

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


        // IMPORTANT: avoid out of bounds memory accesses, which cause no errors but weird bugs.
        if (col >= n_channels) return;

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


def _get_lfilter_fun(b, a, is_fortran=True, axis=0, reverse=False):
    assert axis == 0, "Only filtering along the first axis is currently supported."

    b = np.atleast_1d(b).astype(np.float32)
    a = np.atleast_1d(a).astype(np.float32)
    N = max(len(b), len(a))
    if len(b) < N:
        b = np.pad(b, (0, (N - len(b))), mode='constant')
    if len(a) < N:
        a = np.pad(a, (0, (N - len(a))), mode='constant')
    assert len(a) == len(b)
    kernel = get_lfilter_kernel(N - 1, is_fortran, reverse=reverse)

    lfilter = make_kernel(kernel, 'lfilter', b=b, a=a)

    return lfilter


def _apply_lfilter(lfilter_fun, arr):
    assert isinstance(arr, cp.ndarray)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    n_samples, n_channels = arr.shape

    block = (min(128, n_channels),)
    grid = (int(ceil(n_channels / float(block[0]))),)

    arr = cp.asarray(arr, dtype=np.float32)
    y = cp.zeros_like(arr, order='F' if arr.flags.f_contiguous else 'C', dtype=arr.dtype)

    assert arr.dtype == np.float32
    assert y.dtype == np.float32
    assert arr.shape == y.shape

    lfilter_fun(grid, block, (arr, y, int(y.shape[0]), int(y.shape[1])))
    return y


def lfilter(b, a, arr, axis=0, reverse=False):
    """Perform a linear filter along the first axis on a GPU array."""
    lfilter_fun = _get_lfilter_fun(
        b, a, is_fortran=arr.flags.f_contiguous, axis=axis, reverse=reverse)
    return _apply_lfilter(lfilter_fun, arr)


# GPU FFT-based convolution
# -------------------------

def _clip(x, a, b):
    return max(a, min(b, x))


def pad(fcn_convolve):
    @wraps(fcn_convolve)
    def function_wrapper(x, b, axis=0, **kwargs):
        # add the padding to the array
        xsize = x.shape[axis]
        if 'pad' in kwargs and kwargs['pad']:
            npad = b.shape[axis] // 2
            padd = cp.take(x, cp.arange(npad), axis=axis) * 0
            if kwargs['pad'] == 'zeros':
                x = cp.concatenate((padd, x, padd), axis=axis)
            if kwargs['pad'] == 'constant':
                x = cp.concatenate((padd * 0 + cp.mean(x[:npad]), x, padd + cp.mean(x[-npad:])),
                                   axis=axis)
            if kwargs['pad'] == 'flip':
                pad_in = cp.flip(cp.take(x, cp.arange(1, npad + 1), axis=axis), axis=axis)
                pad_out = cp.flip(cp.take(x, cp.arange(xsize - npad - 1, xsize - 1),
                                          axis=axis), axis=axis)
                x = cp.concatenate((pad_in, x, pad_out), axis=axis)
        # run the convolution
        y = fcn_convolve(x, b, **kwargs)
        # remove padding from both arrays (necessary for x ?)
        if 'pad' in kwargs and kwargs['pad']:
            # remove the padding
            y = cp.take(y, cp.arange(npad, x.shape[axis] - npad), axis=axis)
            x = cp.take(x, cp.arange(npad, x.shape[axis] - npad), axis=axis)
            assert xsize == x.shape[axis]
            assert xsize == y.shape[axis]
        return y
    return function_wrapper


def convolve_cpu(x, b):
    """CPU convolution based on scipy.signal."""
    x = np.asarray(x)
    b = np.asarray(b)
    if b.ndim == 1:
        b = b[:, np.newaxis]
    assert b.ndim == 2
    y = ss.convolve(x, b, mode='same')
    return y


@pad
def convolve_gpu_direct(x, b, **kwargs):
    """Straight GPU FFT-based convolution that fits in memory."""
    if not isinstance(x, cp.ndarray):
        x = np.asarray(x)
    if not isinstance(x, cp.ndarray):
        b = np.asarray(b)
    assert b.ndim == 1
    n = x.shape[0]
    xf = cp.fft.rfft(x, axis=0, n=n)
    if xf.shape[0] > b.shape[0]:
        bp = cp.pad(b, (0, n - b.shape[0]), mode='constant')
        bp = cp.roll(bp, - b.size // 2 + 1)
    else:
        bp = b
    bf = cp.fft.rfft(bp, n=n)[:, np.newaxis]
    y = cp.fft.irfft(xf * bf, axis=0, n=n)
    return y


DEFAULT_CONV_CHUNK = 10_000


@pad
def convolve_gpu_chunked(x, b, pad='flip', nwin=DEFAULT_CONV_CHUNK, ntap=500, overlap=2000):
    """Chunked GPU FFT-based convolution for large arrays.

    This memory-controlled version splits the signal into chunks of n samples.
    Each chunk is tapered in and out, the overlap is designed to get clear of the taper
    splicing of overlaping chunks is done in a cosine way.

    param: pad None, 'zeros', 'constant', 'flip'

    """
    x = cp.asarray(x)
    b = cp.asarray(b)
    assert b.ndim == 1
    n = x.shape[0]
    assert overlap >= 2 * ntap
    # create variables, the gain is to control the splicing
    y = cp.zeros_like(x)
    gain = cp.zeros(n)
    # compute tapers/constants outside of the loop
    taper_in = (-cp.cos(cp.linspace(0, 1, ntap) * cp.pi) / 2 + 0.5)[:, cp.newaxis]
    taper_out = cp.flipud(taper_in)
    assert b.shape[0] < nwin < n
    # this is the convolution wavelet that we shift to be 0 lag
    bp = cp.pad(b, (0, nwin - b.shape[0]), mode='constant')
    bp = cp.roll(bp, - b.size // 2 + 1)
    bp = cp.fft.rfft(bp, n=nwin)[:, cp.newaxis]
    # this is used to splice windows together: cosine taper. The reversed taper is complementary
    scale = cp.minimum(cp.maximum(0, cp.linspace(-0.5, 1.5, overlap - 2 * ntap)), 1)
    splice = (-cp.cos(scale * cp.pi) / 2 + 0.5)[:, cp.newaxis]
    # loop over the signal by chunks and apply convolution in frequency domain
    first = 0
    while True:
        first = min(n - nwin, first)
        last = min(first + nwin, n)
        # the convolution
        x_ = cp.copy(x[first:last, :])
        x_[:ntap] *= taper_in
        x_[-ntap:] *= taper_out
        x_ = cp.fft.irfft(cp.fft.rfft(x_, axis=0, n=nwin) * bp, axis=0, n=nwin)
        # this is to check the gain of summing the windows
        tt = cp.ones(nwin)
        tt[:ntap] *= taper_in[:, 0]
        tt[-ntap:] *= taper_out[:, 0]
        # the full overlap is outside of the tapers: we apply a cosine splicing to this part only
        if first > 0:
            full_overlap_first = first + ntap
            full_overlap_last = first + overlap - ntap
            gain[full_overlap_first:full_overlap_last] *= (1. - splice[:, 0])
            gain[full_overlap_first:full_overlap_last] += tt[ntap:overlap - ntap] * splice[:, 0]
            gain[full_overlap_last:last] = tt[overlap - ntap:]
            y[full_overlap_first:full_overlap_last] *= (1. - splice)
            y[full_overlap_first:full_overlap_last] += x_[ntap:overlap - ntap] * splice
            y[full_overlap_last:last] = x_[overlap - ntap:]
        else:
            y[first:last, :] = x_
            gain[first:last] = tt
        if last == n:
            break
        first += nwin - overlap
    return y


def convolve_gpu(x, b, **kwargs):
    n = x.shape[0]
    # Default chunk size : N samples along the first axis, the one to be chunked and over
    # which to compute the convolution.
    nwin = kwargs.get('nwin', DEFAULT_CONV_CHUNK)
    assert nwin >= 0
    if n <= nwin or nwin == 0:
        return convolve_gpu_direct(x, b)
    else:
        nwin = max(nwin, b.shape[0] + 1)
        return convolve_gpu_chunked(x, b, **kwargs)


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


def svdecon_cpu(X):
    U, S, V = np.linalg.svd(cp.asnumpy(X))
    return U, np.diag(S), V


def free_gpu_memory():
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


# Work around CuPy bugs and limitations
# -----------------------------------------------------------------------------

def mean(x, axis=0):
    if x.ndim == 1:
        return cp.mean(x) if x.size else cp.nan
    else:
        s = list(x.shape)
        del s[axis]
        return (
            cp.mean(x, axis=axis) if x.shape[axis] > 0
            else cp.zeros(s, dtype=x.dtype, order='F'))


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


def var(x):
    return cp.var(x, ddof=1) if x.size > 0 else cp.nan


def ones(shape, dtype=None, order=None):
    # HACK: cp.ones() has no order kwarg at the moment !
    x = cp.zeros(shape, dtype=dtype, order=order)
    x.fill(1)
    return x


def zscore(a, axis=0):
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=0)
    return (a - mns) / sstd
