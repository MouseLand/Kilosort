import numpy as np
import pytest
import torch

from kilosort import run_kilosort


# Use `pytest --runslow` option to include this in tests.
@pytest.mark.slow
def test_pipeline(data_directory, results_directory, saved_ops, torch_device, capture_mgr):
    bin_file = data_directory / 'ZFM-02370_mini.imec0.ap.short.bin'
    with pytest.raises(ValueError):
        # Should result in an error, since `n_chan_bin` isn't specified.
        ops, st, clu, _, _, _, _, _, kept_spikes = run_kilosort(
            settings={}, filename=bin_file, device=torch_device,
            probe_name='neuropixPhase3B1_kilosortChanMap.mat',
            )

    with capture_mgr.global_and_fixture_disabled():
        print('\nStarting run_kilosort test...')
        ops, st, clu, _, _, _, _, _, kept_spikes = run_kilosort(
            filename=bin_file, device=torch_device,
            settings={'n_chan_bin': 385},
            probe_name='neuropixPhase3B1_kilosortChanMap.mat',
            )

    st = st[kept_spikes,0]  # only first column is spike times
    clu = clu[kept_spikes]
        
    # Check that spike times and spike cluster assignments match
    st_load = np.load(results_directory / 'spike_times.npy')
    clu_load = np.load(results_directory / 'spike_clusters.npy')
    saved_yblk = saved_ops['yblk']
    saved_dshift = saved_ops['dshift']
    saved_iKxx = saved_ops['iKxx']

    # Datashift output
    # assert np.allclose(saved_yblk, ops['yblk'])
    # TODO: Why is this resulting in small deviations on different systems?
    # assert np.allclose(saved_dshift, ops['dshift'])
    # TODO: Why is this suddenly getting a dimension mismatch?
    # assert torch.allclose(saved_iKxx, ops['iKxx'])

    # Final spike/neuron readout
    # Less than 2.5% difference in spike count, 5% difference in number of units
    # TODO: Make sure these are reasonable error bounds
    spikes_error = np.abs(st.size - st_load.size)/np.max([st.size, st_load.size])
    with capture_mgr.global_and_fixture_disabled():
        print(f'Proportion difference in total spike count: {spikes_error}')
        print(f'Count from run_kilosort: {st.size}')
        print(f'Count from saved test results: {st_load.size}')

    n = np.unique(clu).size
    n_load = np.unique(clu_load).size
    unit_count_error = np.abs(n - n_load)/np.max([n, n_load])
    with capture_mgr.global_and_fixture_disabled():
        print(f'Proportion difference in number of units: {unit_count_error}')
        print(f'Number of units from run_kilosort: {n}')
        print(f'Number of units from saved test results: {n_load}')

    assert spikes_error <= 0.025
    assert unit_count_error <= 0.05
