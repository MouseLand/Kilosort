import pytest
import tempfile
from pathlib import Path

import numpy as np
import torch

from kilosort import io


def test_probe_io():
    # Create one-column probe with 5 contacts, spaced 1um apart.
    json_probe = {
        'chanMap': np.arange(5),
        'xc': np.ones(5),
        'yc': np.arange(5),
        'kcoords': np.zeros(5),
        'n_chan': 5
    }
    # Repeat in .prb format
    prb_probe = """
channel_groups = {
    0: {
            'channels' : [0,1,2,3,4],
            'geometry': {
                0: [1, 0],
                1: [1, 1],
                2: [1, 2],
                3: [1, 3],
                4: [1, 4]
            }
    }
}
"""
    
    # Save both to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = Path(f.name)
        print(json_file)
        io.save_probe(json_probe, json_file)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.prb', delete=False) as f:
        f.write(prb_probe)
        prb_file = Path(f.name)

    # Load both with kilosort.io
    probe1 = io.load_probe(json_file)
    probe2 = io.load_probe(prb_file)

    print('probe1:')
    print(probe1)
    print('probe2:')
    print(probe2)

    try:
        # Verify that loaded probes contain the same information
        for k in ['chanMap', 'xc', 'yc', 'kcoords', 'n_chan']:
            print(f'testing key {k}')
            assert (k in probe1) and (k in probe2)
            assert np.all(probe1[k] == probe2[k])
    finally:
        # Remove temporary files
        json_file.unlink()
        prb_file.unlink()


def test_bad_channels():
    probe = {
        'xc': np.zeros(5), 'yc': np.arange(5)*10, 'kcoords': np.zeros(5),
        'chanMap': np.arange(5), 'n_chan': 5
        }
    bad_channels = [2, 4]
    probe2 = io.remove_bad_channels(probe, bad_channels)
    assert probe2['n_chan'] == 3
    for k in ['xc', 'yc', 'kcoords']:
        assert probe2[k].size == 3
    assert np.all(probe2['chanMap'] == [0, 1, 3])
    assert np.all(probe['chanMap'] == [0, 1, 2, 3, 4])  # original unchanged

    with pytest.raises(IndexError):
        # These are not in the channel map.
        _ = io.remove_bad_channels(probe, [5, 6])


def test_bat_extension(torch_device, data_directory):
    # Create memmap, write to file, close the file again.
    path = data_directory / 'binary_test' / 'temp_memmap.bat'
    path.parent.mkdir(parents=True, exist_ok=True)
    N, C = (1000, 10)
    r = np.random.rand(N,C)*2 - 1    # scale to (-1,1)
    r = (r*(2**14)).astype(np.int16)     # scale up

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=np.int16)
        a[:] = r[:]
        a.flush()
        del(a)

        directory = path.parent
        filename = io.find_binary(directory)
        assert filename == path
        bfile = io.BinaryFiltered(filename, C, device=torch_device)
        x = bfile[0:100]  # Test data retrieval

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_dat_extension(torch_device, data_directory):
    # Create memmap, write to file, close the file again.
    path = data_directory / 'binary_test' / 'temp_memmap.dat'
    path.parent.mkdir(parents=True, exist_ok=True)
    N, C = (1000, 10)
    r = np.random.rand(N,C)*2 - 1    # scale to (-1,1)
    r = (r*(2**14)).astype(np.int16)     # scale up

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=np.int16)
        a[:] = r[:]
        a.flush()
        del(a)

        directory = path.parent
        filename = io.find_binary(directory)
        assert filename == path
        bfile = io.BinaryFiltered(filename, C, device=torch_device)
        x = bfile[0:100]  # Test data retrieval

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_tmin_tmax(torch_device, data_directory):
    N, C = (1000, 10)
    NT = 300
    nt = 61
    data = np.repeat(np.arange(N)[...,np.newaxis], repeats=C, axis=1)
    fs = 10
    path = data_directory / 'time_interval_test' / 'temp_memmap.dat'
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=np.int16)
        a[:] = data[:]
        a.flush()
        del(a)

        bfile = io.BinaryRWFile(path, n_chan_bin=C, fs=fs, device=torch_device,
                                tmin=10, tmax=85, NT=NT, nt=nt)

        assert bfile.imin == 100
        assert bfile.imax == 850
        assert bfile.n_samples == 750
        assert bfile[0:50].min() == 100
        assert bfile[700:750].max() == 849
        assert bfile.n_batches == 3

        X0 = bfile.padded_batch_to_torch(ibatch=0)
        assert X0.min() == 100
        assert X0.max() == 100 + NT + nt - 1

        X1 = bfile.padded_batch_to_torch(ibatch=1)
        assert X1.min() == 100 + NT - nt
        assert X1.max() == 100 + 2*NT + nt - 1

        X2 = bfile.padded_batch_to_torch(ibatch=2)
        assert X2.min() == 100 + 2*NT - nt
        assert X2.max() == 849

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_tmin_only(torch_device, data_directory):
    N, C = (1000, 10)
    NT = 300
    nt = 61
    data = np.repeat(np.arange(N)[...,np.newaxis], repeats=C, axis=1)
    fs = 10
    path = data_directory / 'time_interval_test' / 'temp_memmap2.dat'
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=np.int16)
        a[:] = data[:]
        a.flush()
        del(a)

        bfile = io.BinaryRWFile(path, n_chan_bin=C, fs=fs, device=torch_device,
                                tmin=43, NT=NT, nt=nt)

        assert bfile.imin == 430
        assert bfile.imax == 1000
        assert bfile.n_samples == 570
        assert bfile[0:10].min() == 430
        assert bfile[400:].max() == 999
        assert bfile.n_batches == 2

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_tmax_only(torch_device, data_directory):
    N, C = (1000, 10)
    NT = 300
    nt = 61
    data = np.repeat(np.arange(N)[...,np.newaxis], repeats=C, axis=1)
    fs = 10
    path = data_directory / 'time_interval_test' / 'temp_memmap3.dat'
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=np.int16)
        a[:] = data[:]
        a.flush()
        del(a)

        bfile = io.BinaryRWFile(path, n_chan_bin=C, fs=fs, device=torch_device,
                                tmax=78, NT=NT, nt=nt)

        assert bfile.imin == 0
        assert bfile.imax == 780
        assert bfile.n_samples == 780
        assert bfile[0:100].min() == 0
        assert bfile[700:].max() == 779
        assert bfile.n_batches == 3

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_file_group(torch_device, data_directory, bfile):
    file = data_directory / 'ZFM-02370_mini.imec0.ap.short.bin'
    fs = bfile.fs
    n_chans = bfile.n_chan_bin

    # Test with file_objects option
    # Load as three 15-second files instead of one 45-second file.
    objs = [np.memmap(file, dtype='int16', shape=bfile.shape, mode='r')
            for _ in range(3)]
    objs[0] = objs[0][:int(15*fs),:]
    objs[1] = objs[1][int(15*fs):int(30*fs),:]
    objs[2] = objs[2][int(30*fs):,:]
    bfg = io.BinaryFileGroup(file_objects=objs)
    bfile2 = io.BinaryFiltered(
        filename='test', n_chan_bin=n_chans, fs=fs, chan_map=bfile.chan_map,
        device=torch_device, file_object=bfg, dtype='int16'
        )

    # First batch, overlapping batch, and last batch
    # (assumes 45s test dataset with 2s batch size)
    for i in [0, 7, 22]:
        b1 = bfile.padded_batch_to_torch(i, skip_preproc=True)
        b2 = bfile2.padded_batch_to_torch(i)
        assert torch.allclose(b1, b2)

    # Test with filenames option
    files = [file]*3  # Load the same data three times
    bfile3 = io.BinaryFiltered(
        filename=files, n_chan_bin=n_chans, fs=fs, chan_map=bfile.chan_map,
        device=torch_device, dtype='int16'
    )

    # First and first, last and last, last of original and last of concat
    for i,j in [(0,0), (21,21), (22,67)]:
        b1 = bfile.padded_batch_to_torch(i, skip_preproc=True)
        b2 = bfile3.padded_batch_to_torch(j)
        assert torch.allclose(b1, b2)
