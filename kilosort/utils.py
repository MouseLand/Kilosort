import os, tempfile, shutil, pathlib
from tqdm import tqdm
from urllib.request import urlopen
from urllib.error import HTTPError

_DOWNLOADS_URL = 'https://www.kilosort.org/downloads'
_DOWNLOADS_DIR_ENV = os.environ.get("KILOSORT_LOCAL_DOWNLOADS_PATH")
_DOWNLOADS_DIR_DEFAULT = pathlib.Path.home().joinpath('.kilosort')
DOWNLOADS_DIR = pathlib.Path(_DOWNLOADS_DIR_ENV) if _DOWNLOADS_DIR_ENV else _DOWNLOADS_DIR_DEFAULT
PROBE_DIR = DOWNLOADS_DIR.joinpath('probes')

# use mat file probes because they enable disconnected channels
probe_names = [
    'neuropixPhase3A_kilosortChanMap.mat',
    'neuropixPhase3B1_kilosortChanMap.mat',
    'neuropixPhase3B2_kilosortChanMap.mat',
    'NP2_kilosortChanMap.mat', 
    'Linear16x1_kilosortChanMap.mat',
    ]

def template_path(basename='wTEMP.npz'):
    """ currently only one set of example templates to use"""
    return cache_template_path(basename)

def cache_template_path(basename):
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    url = f'{_DOWNLOADS_URL}/{basename}'
    cached_file = os.fspath(DOWNLOADS_DIR.joinpath(basename)) 
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, progress=True)
    return cached_file

def download_probes(probe_dir=None):
    if probe_dir is None:
        probe_dir = PROBE_DIR
    probe_dir.mkdir(parents=True, exist_ok=True)
    for probe_name in probe_names:
        url = f'{_DOWNLOADS_URL}/{probe_name}'
        cached_file = os.fspath(probe_dir.joinpath(probe_name)) 
        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(url, cached_file))
            try:
                download_url_to_file(url, cached_file, progress=True)
            except HTTPError as e:
                print(f'Unable to download probe {probe_name}, error:')
                print(e)


def download_url_to_file(url, dst, progress=True):
    r"""Download object at the given URL to a local path.
            Thanks to torch, slightly modified
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
