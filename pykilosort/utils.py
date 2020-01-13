from contextlib import contextmanager
from functools import reduce
import json
import logging
from pathlib import Path
import operator
import os.path as op
import re
from time import perf_counter

import numpy as np
import cupy as cp

from .event import emit, connect, unconnect  # noqa

logger = logging.getLogger(__name__)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


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


def memmap_binary_file(dat_path, n_channels=None, shape=None, dtype=None, offset=None):
    """Memmap a dat file."""
    assert dtype is not None
    item_size = np.dtype(dtype).itemsize
    offset = offset if offset else 0
    if shape is None:
        assert n_channels is not None
        n_samples = (op.getsize(str(dat_path)) - offset) // (item_size * n_channels)
        shape = (n_channels, n_samples)
    assert shape
    shape = tuple(shape)
    return np.memmap(str(dat_path), dtype=dtype, shape=shape, offset=offset, order='F')


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


class LargeArrayWriter(object):
    """Save a large array chunk by chunk, in a binary file with FORTRAN order."""
    def __init__(self, path, dtype=None, shape=None):
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self._shape = shape
        assert shape[-1] == -1  # the last axis must be the extendable axis, in FORTRAN order
        assert -1 not in shape[:-1]  # shape may not contain -1 outside the last dimension
        self.fw = open(self.path, 'wb')
        self.extendable_axis_size = 0
        self.total_size = 0

    def append(self, arr):
        # We convert to the requested data type.
        assert arr.flags.f_contiguous  # only FORTRAN order arrays are currently supported
        assert arr.shape[:-1] == self._shape[:-1]
        arr = arr.astype(self.dtype)
        es = arr.shape[-1]
        if arr.flags.f_contiguous:
            arr = arr.T
        # We download the array from the GPU if required.
        # We ensure the array is in FORTRAN order now.
        assert arr.flags.c_contiguous
        if isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)
        arr.tofile(self.fw)
        self.total_size += arr.size
        self.extendable_axis_size += es  # the last dimension, but
        assert prod(self.shape) == self.total_size

    @property
    def shape(self):
        return self._shape[:-1] + (self.extendable_axis_size,)

    def close(self):
        self.fw.close()
        # Save JSON metadata file.
        with open(self.path.with_suffix('.json'), 'w') as f:
            json.dump({'shape': self.shape, 'dtype': str(self.dtype), 'order': 'F'}, f)


def memmap_large_array(path):
    """Memmap a large array saved by LargeArrayWriter."""
    path = Path(path)
    with open(path.with_suffix('.json'), 'r') as f:
        metadata = json.load(f)
    assert metadata['order'] == 'F'
    dtype = np.dtype(metadata['dtype'])
    shape = metadata['shape']
    return memmap_binary_file(path, shape=shape, dtype=dtype)


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

    def path(self, name, ext='.npy'):
        """Path to an array in the context directory."""
        return self.context_path / (name + ext)

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


def load_probe(probe_path):
    """Load a .mat probe file from Kilosort2, or a PRB file (experimental)."""

    # A bunch with the following attributes:
    _required_keys = ('NchanTOT', 'chanMap', 'xc', 'yc', 'kcoords')
    probe = Bunch()
    probe.NchanTOT = 0
    probe_path = Path(probe_path).resolve()

    if probe_path.suffix == '.prb':
        # Support for PRB files.
        contents = probe_path.read_text()
        metadata = {}
        exec(contents, {}, metadata)
        probe.chanMap = []
        probe.xc = []
        probe.yc = []
        probe.kcoords = []
        for cg in sorted(metadata['channel_groups']):
            d = metadata['channel_groups'][cg]
            ch = d['channels']
            pos = d.get('geometry', {})
            probe.chanMap.append(ch)
            probe.NchanTOT += len(ch)
            probe.xc.append([pos[c][0] for c in ch])
            probe.yc.append([pos[c][1] for c in ch])
            probe.kcoords.append([cg for c in ch])
        probe.chanMap = np.concatenate(probe.chanMap).ravel().astype(np.int32)
        probe.xc = np.concatenate(probe.xc)
        probe.yc = np.concatenate(probe.yc)
        probe.kcoords = np.concatenate(probe.kcoords)

    elif probe_path.suffix == '.mat':
        from scipy.io import loadmat
        mat = loadmat(probe_path)
        probe.xc = mat['xcoords'].ravel().astype(np.float64)
        nc = len(probe.xc)
        probe.yc = mat['ycoords'].ravel().astype(np.float64)
        probe.kcoords = mat.get('kcoords', np.zeros(nc)).ravel().astype(np.float64)
        probe.chanMap = (mat['chanMap'] - 1).ravel().astype(np.int32)  # NOTE: 0-indexing in Python
        probe.NchanTOT = len(probe.chanMap)  # NOTE: should match the # of columns in the raw data

    for n in _required_keys:
        assert n in probe.keys()

    return probe
