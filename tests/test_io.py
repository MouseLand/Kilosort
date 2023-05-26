import json
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
