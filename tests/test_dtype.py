import pytest
from pathlib import Path

import numpy as np
import torch

from kilosort import io


data_shape = (1000, 10)

def make_bfile_with_dtype(dtype, fill, directory, device):
    # Create memmap, write to file, close the file again.
    N, C = fill.shape
    path = directory / 'temp_memmap.bin'

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=dtype)
        a[:] = fill[:]
        a.flush()
        del(a)

        bfile = io.BinaryFiltered(path, C, device=device, dtype=dtype)
        x = bfile[0:100]
        y = bfile.padded_batch_to_torch(0)

        # Memmap should be loaded as the specified dtype, but the Tensor used for
        # computations in BinaryFiltered should always be converted to float32.
        assert bfile.file.dtype == dtype
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

        bfile.close()

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_uint16(torch_device, data_directory):
    d = 'uint16'
    r = np.random.rand(*data_shape)*(2**30)  # scale to nearly max value
    r = r.astype(d)
    make_bfile_with_dtype(d, r, data_directory, torch_device)

def test_int16(torch_device, data_directory):
    d = 'int16'
    r = np.random.rand(*data_shape)*2 - 1    # scale to (-1,1)
    r = (r*(2**14)).astype(d)                # scale to nearly max value
    make_bfile_with_dtype(d, r, data_directory, torch_device)

def test_int32(torch_device, data_directory):
    d = 'int32'
    r = np.random.rand(*data_shape)*2 - 1    # scale to (-1,1)
    r = (r*(2**30)).astype(d)                # scale to nearly max value
    make_bfile_with_dtype(d, r, data_directory, torch_device)

def test_float32(torch_device, data_directory):
    d = 'float32'
    r = np.random.rand(*data_shape)*2 - 1    # scale to (-1,1)
    r = (r*(2**15)).astype(d)                # scale up, but keep precision
    make_bfile_with_dtype(d, r, data_directory, torch_device)

def test_unsupported(torch_device, data_directory):
    for d in ['uint8', 'int8', 'float64']:
        # May or may not cause an error, but should always warn user that these
        # dtypes are not specifically supported and might cause unexpected
        # behavior.
        with pytest.warns(RuntimeWarning):
            r = (np.random.rand(*data_shape)*(2**6)).astype(d)
            try:
                make_bfile_with_dtype(d, r, data_directory, torch_device)
            except:
                # Unsupported dtypes might cause unexpected errors, that's okay
                # as long as the warning is raised first.
                continue
