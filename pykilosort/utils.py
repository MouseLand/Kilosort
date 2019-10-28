from pathlib import Path
import os.path as op
import re

import numpy as np
# import cupy as cp

from .event import emit, connect, unconnect  # noqa


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return Bunch(super(Bunch, self).copy())


def p(x):
    print("shape", x.shape, "mean", "%5e" % x.mean())
    print(x[:2, :2])
    print()
    print(x[-2:, -2:])


def is_fortran(x):
    if isinstance(x, np.ndarray):
        return x.flags.f_contiguous


def read_data(dat_path, offset=0, shape=None, dtype=None, axis=0):
    count = shape[0] * shape[1] if shape and -1 not in shape else -1
    buff = np.fromfile(dat_path, dtype=dtype, count=count, offset=offset)
    if shape and -1 not in shape:
        shape = (-1, shape[1]) if axis == 0 else (shape[0], -1)
    if shape:
        buff = buff.reshape(shape, order='F')
    return buff


def memmap_raw_data(dat_path, n_channels=None, dtype=None, offset=None, order=None):
    """Memmap a dat file."""
    assert dtype is not None
    assert n_channels is not None
    item_size = np.dtype(dtype).itemsize
    offset = offset if offset else 0
    n_samples = (op.getsize(str(dat_path)) - offset) // (item_size * n_channels)
    return np.memmap(
        str(dat_path), dtype=dtype, shape=(n_samples, n_channels), offset=offset, order=order)


def extract_constants_from_cuda(code):
    r = re.compile(r'const int\s+\S+\s+=\s+\S+.+')
    m = r.search(code)
    if m:
        constants = m.group(0).replace('const int', '').replace(';', '').split(',')
        for const in constants:
            a, b = const.strip().split('=')
            yield a.strip(), int(b.strip())


def get_cuda(fn):
    path = Path(__file__).parent / 'cuda' / (fn + '.cu')
    assert path.exists
    code = path.read_text()
    code = code.replace('__global__ void', 'extern "C" __global__ void')
    return code, Bunch(extract_constants_from_cuda(code))
