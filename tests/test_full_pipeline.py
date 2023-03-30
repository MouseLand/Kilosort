from pathlib import Path

import numpy as np
import pytest

from kilosort import run_kilosort, default_settings


# Use `pytest --runslow` option to include this in tests.
@pytest.mark.slow
def test_pipeline():
    settings = default_settings()
    # TODO: replace this with a pytest fixture that downloads test data from
    #       remote repository. 
    settings['data_dir'] = "C:/code/kilosort4_data/"
    # TODO: add option to not save results to file? Otherwise this will need
    #       to store and overwrite some results files every time, which is
    #       unnecessary.
    _, st, clu, _, _, _, _, _ = run_kilosort(
        settings=settings, probe_name='neuropixPhase3B1_kilosortChanMap.mat'
        )
    st = st[:,0]  # only first column is spike times, that's all that gets saved
        
    # Check that spike times and spike cluster assignments match
    results_dir = Path(settings['data_dir']).joinpath('pytest/')
    st_load = np.load(results_dir / 'spike_times.npy')
    clu_load = np.load(results_dir / 'spike_clusters.npy')

    assert np.allclose(st, st_load)
    assert np.allclose(clu, clu_load)
    # TODO: What else? Or is that sufficient for now?
    

def test_step_by_step():
    # TODO: As above, but break out into individual steps with result of each
    #       operation pre-saved, so that if there's a mismatch it's easy to
    #       identify where it's coming from.
    pass
