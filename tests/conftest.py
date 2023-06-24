from pathlib import Path
import shutil

import pytest
import torch

from kilosort import io
from kilosort.utils import download_probes


@pytest.fixture(scope='session')
def download(request):
    return request.config.getoption('--download')

@pytest.fixture(scope='session')
def torch_device():
        return torch.device('cpu')


### runslow flag configured according to response from Manu CJ here:
# https://stackoverflow.com/questions/47559524/pytest-how-to-skip-tests-unless-you-declare-an-option-flag
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--download", action="store", default='', help="binary, results, or both"
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

### End


### Configure data paths, download data if not already present.
# Adapted from https://github.com/MouseLand/suite2p/blob/main/conftest.py
@pytest.fixture(scope='session')
def data_directory(download):
    """Specifies directory for test data and results, downloads if needed."""

    # Set path to directory within tests/ folder dynamically
    data_path = Path.home() / '.kilosort/.test_data/'
    data_path.mkdir(parents=True, exist_ok=True)

    binary_path = data_path / 'ZFM-02370_mini.imec0.ap.bin'
    binary_url = 'https://www.kilosort.org/downloads/ZFM-02370_mini.imec0.ap.zip'
    if (download == 'binary') or (download == 'both'):
        if binary_path.is_file():
            binary_path.unlink()
    if not binary_path.is_file():
        download_data(binary_path, binary_url)

    results_path = data_path / 'saved_results/'
    results_url = 'https://www.kilosort.org/downloads/pytest.zip'
    if ((download == 'results') or (download == 'both')):
        if results_path.is_dir():
            shutil.rmtree(results_path)
    if not results_path.is_dir():
        # Downloaded folder extracts here, get rid of any existing results
        # from running tests.
        ks_path = data_path / 'Kilosort4'
        if ks_path.is_dir():
            shutil.rmtree(ks_path)
        download_data(results_path, results_url)
        # Rename folder
        shutil.move(ks_path, results_path)

    # Download default probe files if they don't already exist.
    download_probes()

    return data_path


def download_data(local, remote):
    """Download and unzip `remote` data to `local` path."""
    # Lazy import to reduce test overhead when files are already downloaded
    import zipfile

    zip_file = local.with_suffix('.zip')
    download_url_to_file(remote, zip_file)  
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(local.parent)
    zip_file.unlink()  # delete zip archive after extracting data

# TODO: look at tenacity package, determine if this is necessary, would introduce
#       another dependency
# @retry
def download_url_to_file(url, dst, progress=True):
    """Download object at the given URL to a local path.
    
    Parameters
    ----------
    url: str
        URL of the object to download
    dst: str
        Full path where object will be saved, e.g. `/tmp/temporary_file`
    progress: bool, default=True.
        Whether or not to display a progress bar to stderr.

    """

    # Lazy imports to reduce test overhead when files are already downloaded.
    import ssl
    from urllib.request import urlopen
    import tempfile
    from tqdm import tqdm
    import shutil

    ssl._create_default_https_context = ssl._create_unverified_context
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    else:
        file_size = None

    # We deliberately save to a temp file and move it after
    # TODO: explain why
    dst = dst.expanduser()
    dst_dir = dst.parent
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    print(f"\nDownloading: {url}")

    try:
        with tqdm(total=file_size, disable=(not progress),
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        # Move from temporary file to specified destination.
        shutil.move(f.name, dst)
    finally:
        # Close and delete temporary file
        f.close()
        p = Path(f.name)
        if p.is_file():
            p.unlink()


@pytest.fixture(scope='session')
def results_directory(data_directory):
    return data_directory / "saved_results"

### End


@pytest.fixture(scope='session')
def saved_ops(results_directory, torch_device):
    ops = io.load_ops(results_directory / 'ops.npy', device=torch_device)
    return ops

@pytest.fixture(scope='session')
def bfile(saved_ops, torch_device, data_directory):
    # TODO: add option to load BinaryFiltered from ops dict, move this code
    #       to that function
    settings = saved_ops['settings']
    # Don't get filename from settings, will be different based on OS and which
    # system ran tests originally.
    filename = data_directory / 'ZFM-02370_mini.imec0.ap.bin'

    # TODO: add option to load BinaryFiltered from ops dict, move this code
    #       to that function
    bfile = io.BinaryFiltered(
        filename, settings['n_chan_bin'], settings['fs'],
        settings['NT'], settings['nt'], settings['nt0min'],
        saved_ops['probe']['chanMap'], hp_filter=saved_ops['fwav'],
        whiten_mat=saved_ops['Wrot'], dshift=saved_ops['dshift'],
        device=torch_device
        )
    
    return bfile

### End
