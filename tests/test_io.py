import pytest
import tempfile
from pathlib import Path

import numpy as np

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

        bfile.close()

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

        bfile.close()

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

        bfile.close()

    finally:
        # Delete memmap file and re-raise exception
        bfile.close()
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

        bfile.close()

    finally:
        # Delete memmap file and re-raise exception
        bfile.close()
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

        bfile.close()

    finally:
        # Delete memmap file and re-raise exception
        bfile.close()
        path.unlink()
