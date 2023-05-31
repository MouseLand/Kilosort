from pathlib import Path

import numpy as np
import pytest
import torch

from kilosort import run_kilosort, default_settings


# Use `pytest --runslow` option to include this in tests.
@pytest.mark.slow
def test_pipeline(data_directory, results_directory, saved_ops, torch_device):

    ops, st, clu, _, _, _, _, _ = run_kilosort(
        data_dir=data_directory, probe_name='neuropixPhase3B1_kilosortChanMap.mat',
        device=torch_device
        )
    st = st[:,0]  # only first column is spike times, that's all that gets saved
        
    # Check that spike times and spike cluster assignments match
    st_load = np.load(results_directory / 'spike_times.npy')
    clu_load = np.load(results_directory / 'spike_clusters.npy')
    saved_yblk = saved_ops['yblk']
    saved_dshift = saved_ops['dshift']
    saved_iKxx = saved_ops['iKxx'].to(torch_device)

    # Datashift output
    assert np.allclose(saved_yblk, ops['yblk'])
    assert np.allclose(saved_dshift, ops['dshift'])
    assert torch.allclose(saved_iKxx, ops['iKxx'])
    # Final spike/neuron readout
    assert np.allclose(st, st_load)
    assert np.allclose(clu, clu_load)
    # TODO: What else? Or is that sufficient for now?
    