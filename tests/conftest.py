from pathlib import Path

import numpy as np
import pytest
import torch

import kilosort.preprocessing as kpp
from kilosort import io
from kilosort import PROBE_DIR


# TODO: rename this so that flag is something more informative like
#       "--regression"
# runslow flag configured according to response from Manu CJ here:
# https://stackoverflow.com/questions/47559524/pytest-how-to-skip-tests-unless-you-declare-an-option-flag
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture()
def torch_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# TODO: have a short sample version and a full version
# TODO: also download probe file if not already present
@pytest.fixture()
def data_directory():
    """Specifies directory for test data and pre-computed results."""
    data_path = Path("C:/code/kilosort4_data/")
    # TODO: move this to a remote repository, hard-coded for Jacob's PC for now.

    return data_path

@pytest.fixture()
def results_directory(data_directory):
    return data_directory.joinpath("pytest/")

@pytest.fixture()
def saved_ops(results_directory):
    ops = np.load(results_directory / 'ops.npy', allow_pickle=True).item()
    return ops

@pytest.fixture()
def bfile(saved_ops, torch_device):
    # TODO: add option to load BinaryFiltered from ops dict, move this code
    #       to that function
    settings = saved_ops['settings']

    bfile = io.BinaryFiltered(
        settings['filename'], settings['n_chan_bin'], settings['fs'],
        settings['NT'], settings['nt'], settings['nt0min'],
        saved_ops['probe']['chanMap'], hp_filter=saved_ops['fwav'],
        whiten_mat=saved_ops['Wrot'], dshift=saved_ops['dshift'],
        device=torch_device
        )
    
    return bfile
