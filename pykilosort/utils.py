from contextlib import contextmanager
import json
import logging
from pathlib import Path
import os.path as op
import re
from time import perf_counter

import numpy as np
import cupy as cp

from .event import emit, connect, unconnect  # noqa

logger = logging.getLogger(__name__)


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


def _extend(x, i0, i1, val, axis=0):
    """Extend an array along a dimension and fill it with some values."""
    shape = x.shape
    if x.shape[axis] < i1:
        s = list(x.shape)
        s[axis] = i1 - s[axis]
        x = cp.concatenate((x, cp.zeros(tuple(s), dtype=x.dtype, order='F')), axis=axis)
        assert x.shape[axis] == i1
    s = [slice(None, None, None)] * x.ndim
    s[axis] = slice(i0, i1, 1)
    x[s] = val
    for i in range(x.ndim):
        if i != axis:
            assert x.shape[i] == shape[i]
    return x


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


def memmap_raw_data(dat_path, n_channels=None, dtype=None, offset=None):
    """Memmap a dat file."""
    assert dtype is not None
    assert n_channels is not None
    item_size = np.dtype(dtype).itemsize
    offset = offset if offset else 0
    n_samples = (op.getsize(str(dat_path)) - offset) // (item_size * n_channels)
    return np.memmap(
        str(dat_path), dtype=dtype, shape=(n_channels, n_samples), offset=offset, order='F')


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


class Context(Bunch):
    def __init__(self, context_path):
        super(Context, self).__init__()
        self.context_path = context_path
        self.intermediate = Bunch()
        self.context_path.mkdir(exist_ok=True, parents=True)
        self.timer = {}

    @property
    def metadata_path(self):
        return self.context_path / 'metadata.json'

    def path(self, name):
        """Path to an array in the context directory."""
        return self.context_path / (name + '.npy')

    def read_metadata(self):
        """Read the metadata dictionary from the metadata.json file in the context dir."""
        if not self.metadata_path.exists():
            return Bunch()
        with open(self.metadata_path, 'r') as f:
            return Bunch(json.load(f))

    def write_metadata(self, metadata):
        """Write metadata dictionary in the metadata.json file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def read(self, name):
        """Read an array from memory (intermediate object) or from disk."""
        if name not in self.intermediate:
            path = self.path(name)
            # Load a NumPy file.
            if path.exists():
                logger.debug("Loading %s.npy", name)
                # Memmap for large files.
                mmap_mode = 'r' if op.getsize(path) > 1e8 else None
                self.intermediate[name] = np.load(path, mmap_mode=mmap_mode)
            else:
                # Load a value from the metadata file.
                self.intermediate[name] = self.read_metadata().get(name, None)
        return self.intermediate[name]

    def write(self, **kwargs):
        """Write several arrays."""
        # Load the metadata.
        if self.metadata_path.exists():
            metadata = self.read_metadata()
        else:
            metadata = Bunch()
        # Write all variables.
        for k, v in kwargs.items():
            # Transfer GPU arrays to the CPU before saving them.
            if isinstance(v, cp.ndarray):
                v = cp.asnumpy(v)
            if isinstance(v, np.ndarray):
                p = self.path(k)
                overwrite = ' (overwrite)' if p.exists() else ''
                logger.debug("Saving %s.npy%s", k, overwrite)
                np.save(p, np.asfortranarray(v))
            elif v is not None:
                logger.debug("Save %s in the metadata.json file.", k)
                metadata[k] = v
        # Write the metadata file.
        self.write_metadata(metadata)

    def load(self):
        """Load intermediate results from disk."""
        # Load metadata values that are not already loaded in the intermediate dictionary.
        self.intermediate.update(
            {k: v for k, v in self.read_metadata().items() if k not in self.intermediate})
        # Load NumPy arrays that are not already loaded in the intermediate dictionary.
        names = [f.stem for f in self.context_path.glob('*.npy')]
        self.intermediate.update(
            {name: self.read(name) for name in names if name not in self.intermediate})

    def save(self, **kwargs):
        """Save intermediate results to the ctx.intermediate dictionary, and to disk also.

        This has two effects:
        1. variables are available via ctx.intermediate in the current session
        2. In a future session with ctx.load(), these variables will be readily available in
           ctx.intermediate

        """
        for k, v in kwargs.items():
            if v is not None:
                self.intermediate[k] = v
        kwargs = kwargs or self.intermediate
        self.write(**kwargs)

    @contextmanager
    def time(self, name):
        """Context manager to measure the time of a section of code."""
        t0 = perf_counter()
        yield
        t1 = perf_counter()
        self.timer[name] = t1 - t0
        self.show_timer(name)

    def show_timer(self, name=None):
        """Display the results of the timer."""
        if name:
            logger.info("Step `{:s}` took {:.2f}s.".format(name, self.timer[name]))
            return
        for name in self.timer.keys():
            self.show_timer(name)
