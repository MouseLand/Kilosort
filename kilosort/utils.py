import os, tempfile, shutil, pathlib, psutil
import importlib.util
import logging
logger = logging.getLogger(__name__)

from tqdm import tqdm
from urllib.request import urlopen
from urllib.error import HTTPError

import torch

# pynvml is an optional dependency for Pytorch cuda, so existing installations
# may or may not have it. It's not worth causing an interruption for this.
if importlib.util.find_spec('pynvml') is not None:
    _NVML_EXISTS = True
else:
    _NVML_EXISTS = False

_DOWNLOADS_URL = 'https://www.kilosort.org/downloads'
_DOWNLOADS_DIR_ENV = os.environ.get("KILOSORT_LOCAL_DOWNLOADS_PATH")
_DOWNLOADS_DIR_DEFAULT = pathlib.Path.home().joinpath('.kilosort')
DOWNLOADS_DIR = pathlib.Path(_DOWNLOADS_DIR_ENV) if _DOWNLOADS_DIR_ENV else _DOWNLOADS_DIR_DEFAULT
PROBE_DIR = DOWNLOADS_DIR.joinpath('probes')

# use mat file probes because they enable disconnected channels
probe_names = [
    'neuropixPhase3A_kilosortChanMap.mat',
    'neuropixPhase3B1_kilosortChanMap.mat',\
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
        logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
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
            logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            try:
                download_url_to_file(url, cached_file, progress=True)
            except HTTPError as e:
                logger.info(f'Unable to download probe {probe_name}, error:')
                logger.info(e)


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


def log_performance(log=None, level=None, header=None):
    """Log usage information for cpu, memory, gpu, and gpu memory.

    Parameters
    ----------
    log : logging.Logger; optional.
        Logger object used to write the text. If not provided, the logger for
        `kilosort.utils` will be used.
    level : str; optional.
        Logging level to use. By default, 'debug' will be used. See documentation
        for the built-in `logging` module for information on logging levels.
    header : str; optional.
        Text to output before usage information. For example, an iteration number
        when logging within a loop.

    Notes
    -----
    Usage values include resources used by non-Kilosort processes. This is done
    so that, in the event of an error or crash, we can see if it was due to
    exceeding system resources regardless of whether Kilosort itself was using
    all of them.

    """

    if log is None:
        log = logger
    if level is None:
        level = 'debug'
    if header is not None:
        getattr(log, level)(' ')
        getattr(log, level)(f'{header}')

    getattr(log, level)('*'*56)
    # TODO: This part is slow, on the order of ms versus micro-seconds for GPU check.
    #       Find a faster way to check main memory and cpu usage.
    getattr(log, level)(f'CPU usage:    {psutil.cpu_percent():5.2f} %')

    memory = psutil.virtual_memory()
    used = memory.used / 2**30
    total = memory.total / 2**30
    pct = (used / total) * 100
    getattr(log, level)(f'Memory:       {pct:5.2f} %     |{used:10.2f}   / {total:8.2f} GB')
    getattr(log, level)('-'*54)

    if torch.cuda.is_available():
        if _NVML_EXISTS:
            getattr(log, level)(f'GPU usage:    {torch.cuda.utilization():5.2f} %')
        else:
            getattr(log, level)(f'GPU usage:    `conda install pynvml` for GPU usage')

        gpu_avail, gpu_total = torch.cuda.mem_get_info()
        gpu_avail /= (2**30)  # convert bytes -> gb
        gpu_total /= (2**30)
        gpu_used = gpu_total - gpu_avail
        gpu_pct = (gpu_used / gpu_total) * 100

        getattr(log, level)(f'GPU memory:   {gpu_pct:5.2f} %     |{gpu_used:10.2f}   / {gpu_total:8.2f} GB')
        allocated = torch.cuda.memory_allocated() / 2**30
        alloc_pct = (allocated / gpu_total) * 100
        getattr(log, level)(f'Allocated:    {alloc_pct:5.2f} %     |{allocated:10.2f}   / {gpu_total:8.2f} GB')
        max_alloc = torch.cuda.max_memory_allocated() / 2**30
        max_pct = (max_alloc / gpu_total) * 100
        getattr(log, level)(f'Max alloc:    {max_pct:5.2f} %     |{max_alloc:10.2f}   / {gpu_total:8.2f} GB')
    else:
        getattr(log, level)('GPU usage:    N/A')
        getattr(log, level)('GPU memory:   N/A')

    getattr(log, level)('*'*56)


def log_cuda_details(log):
    """Log a detailed summary of cuda stats from `torch.cuda.memory_summary`."""
    if torch.cuda.is_available():
        log.debug(f'\n\n{torch.cuda.memory_summary(abbreviated=True)}\n')
