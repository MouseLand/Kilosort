from math import ceil
from .utils import Bunch

default_params = Bunch()

# sample rate
default_params.fs = 30000.

# frequency for high pass filtering (150)
default_params.fshigh = 150.
default_params.fslow = None

# minimum firing rate on a "good" channel (0 to skip)
default_params.minfr_goodchannels = 0.1

# threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
default_params.Th = [10, 4]

# how important is the amplitude penalty (like in Kilosort1, 0 means not used,
# 10 is average, 50 is a lot)
default_params.lam = 10

# splitting a cluster at the end requires at least this much isolation for each sub-cluster (max=1)
default_params.AUCsplit = 0.9

# minimum spike rate (Hz), if a cluster falls below this for too long it gets removed
default_params.minFR = 1. / 50

# number of samples to average over (annealed from first to second value)
default_params.momentum = [20, 400]

# spatial constant in um for computing residual variance of spike
default_params.sigmaMask = 30

# threshold crossings for pre-clustering (in PCA projection space)
default_params.ThPre = 8

# danger, changing these settings can lead to fatal errors
# options for determining PCs
default_params.spkTh = -6  # spike threshold in standard deviations (-6)
default_params.reorder = 1  # whether to reorder batches for drift correction.
default_params.nskip = 25  # how many batches to skip for determining spike PCs

# default_params.GPU = 1  # has to be 1, no CPU version yet, sorry
# default_params.Nfilt = 1024 # max number of clusters
default_params.nfilt_factor = 4  # max number of clusters per good channel (even temporary ones)
default_params.ntbuff = 64  # samples of symmetrical buffer for whitening and spike detection
# must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory).
default_params.whiteningRange = 32  # number of channels to use for whitening each channel
default_params.nSkipCov = 25  # compute whitening matrix from every N-th batch
default_params.scaleproc = 200  # int16 scaling of whitened data
default_params.nPCs = 3  # how many PCs to project the spikes into
# default_params.useRAM = 0  # not yet available

default_params.nt0 = 61
default_params.nup = 10
default_params.sig = 1
default_params.gain = 1

default_params.loc_range = [5, 4]
default_params.long_range = [30, 6]


def set_dependent_params(params):
    """Add dependent parameters."""
    # we need buffers on both sides for filtering
    params.NT = params.get('NT', 64 * 1024 + params.ntbuff)
    params.NTbuff = params.get('NTbuff', params.NT + 4 * params.ntbuff)
    params.nt0min = params.get('nt0min', ceil(20 * params.nt0 / 61))
    return params
