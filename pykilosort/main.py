import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .preprocess import preprocess
from .cluster import clusterSingleBatches
from .learn import learnAndSolve8b
from .postprocess import find_merges, splitAllClusters, set_cutoff, rezToPhy
from .utils import Bunch
from .default_params import default_params

import numpy as np
import cupy as cp

logger = logging.getLogger(__name__)


class Context(Bunch):
    def __init__(self, dir_path):
        super(Context, self).__init__()
        self.dir_path = dir_path
        self.intermediate = Bunch()

    @property
    def context_path(self):
        """Path to the context directory."""
        return self.dir_path / '.kilosort/context/'

    def path(self, name):
        """Path to an array in the context directory."""
        return self.context_path / ('%s.npy' % name)

    def read(self, *names):
        """Read several arrays, from memory (intermediate object) or from disk."""
        out = [self.intermediate.get(name, np.load(self.path(name))) for name in names]
        if len(out) == 1:
            return out[0]

    def write(self, **kwargs):
        """Write several arrays."""
        for k, v in kwargs:
            if isinstance(v, cp.ndarray):
                v = cp.asnumpy(v)
            logger.debug("Saving %s.npy", k)
            np.save(self.path(k), v)

    def load(self):
        """Load intermediate results from disk."""
        names = self.context_path.glob('*.npy')
        self.intermediate.update({name: value for name, value in zip(names, self.read(names))})

    def save(self):
        """Save intermediate results to disk."""
        self.write(**self.intermediate)


def default_probe(raw_data):
    nc = raw_data.shape[1]
    return Bunch(Nchan=nc, xc=np.zeros(nc), yc=np.arange(nc))


def run(dir_path=None, raw_data=None, probe=None, params=None):
    """

    probe has the following attributes:
    - xc
    - yc
    - Nchan

    """

    if dir_path is None:
        raise ValueError("Please provide a dir_path.")
    if raw_data is None:
        raise ValueError("Please provide a raw_data array.")

    dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True, parents=True)
    assert dir_path.exists()

    # Get or create the probe object.
    probe = probe or default_probe(raw_data)
    assert probe

    # Get params.
    user_params = params
    params = default_params.copy()
    params.update(user_params)
    assert params

    # Create the context.
    ctx = Context(dir_path)
    ctx.params = params
    ctx.probe = probe
    ctx.raw_data = raw_data

    # Load the intermediate results to avoid recomputing things.
    ctx.load()
    ir = ctx.intermediate

    # preprocess data to create proc.dat
    proc_path = dir_path / 'proc.dat'
    if not proc_path.exists():
        # Do not preprocess again if the proc.dat file already exists.
        ir.Wrot = preprocess(raw_data=raw_data, probe=probe, params=params, proc_path=proc_path)
    # Get the whitening matrix, from memory if it has already been computed/loaded, or from disk.
    ir.Nbatch, ir.Wrot = ctx.read('Wrot')

    # Open the proc file.
    assert proc_path.exists()
    ir.proc = np.memmap(proc_path, dtype=raw_data.dtype, mode='r', order='F')

    # time-reordering as a function of drift
    # This function adds to the intermediate object: iorig, ccb0, ccbsort
    clusterSingleBatches(ctx)

    # main tracking and template matching algorithm
    learnAndSolve8b(ctx)

    # final merges
    find_merges(ctx, 1)

    # final splits by SVD
    splitAllClusters(ctx, 1)

    # final splits by amplitudes
    splitAllClusters(ctx, 0)

    # decide on cutoff
    set_cutoff(ctx)

    logger.info('Found %d good units.', np.sum(ctx.good > 0))

    # write to Phy
    logger.info('Saving results to Phy.')
    rezToPhy(ctx, rootZ)

    # if you want to save the results to a Matlab file...

    # discard features in final rez file (too slow to save)
    # rez.cProj = []
    # rez.cProjPC = []
