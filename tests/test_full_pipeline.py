from pathlib import Path

import numpy as np
import pytest

from kilosort import run_kilosort, default_settings


# Use `pytest --runslow` option to include this in tests.
@pytest.mark.slow
def test_pipeline():
    settings = default_settings()
    # TODO: replace this with a dedicated test dataset that's stored inside
    #       test directory so that path can be determined dynamically
    settings['data_dir'] = "C:/code/kilosort4_data/"
    # TODO: add option to not save results to file? Otherwise this will need
    #       to overwrite some results files every time, which is unnecessary.
    # ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = \
    #   run_kilosort(settings=settings, probe_name='neuropixPhase3B1_kilosortChanMap.mat')
    _, st, clu, _, _, _, _, _ = run_kilosort(
        settings=settings, probe_name='neuropixPhase3B1_kilosortChanMap.mat',
        save=False
        )
        
    # Check that spike times and spike cluster assignments match
    results_dir = Path(settings['data_dir']).joinpath('kilosort4')
    st_load = np.load(results_dir / 'spike_times.npy')
    clu_load = np.load(results_dir / 'spike_clusters.npy')

    assert np.allclose(st, st_load)
    assert np.allclose(clu, clu_load)
    # TODO: What else? Or is that sufficient?
    

def test_step_by_step():
    # TODO: As above, but break out into individual steps with result of each
    #       operation pre-saved, so that if there's a mismatch it's easy to
    #       identify where it's coming from.
    pass
