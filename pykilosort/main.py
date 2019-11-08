import logging
from pathlib import Path

import numpy as np

from .preprocess import preprocess, get_good_channels, get_whitening_matrix, get_Nbatch
from .cluster import clusterSingleBatches
from .learn import learnAndSolve8b
from .postprocess import find_merges, splitAllClusters, set_cutoff, rezToPhy
from .utils import Bunch, Context
from .default_params import default_params, set_dependent_params

logger = logging.getLogger(__name__)


def default_probe(raw_data):
    nc = raw_data.shape[1]
    return Bunch(Nchan=nc, xc=np.zeros(nc), yc=np.arange(nc))


def run(dir_path=None, raw_data=None, probe=None, params=None, dat_path=None):
    """

    probe has the following attributes:
    - xc
    - yc
    - kcoords
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
    user_params = params or {}
    params = default_params.copy()
    params.update(user_params)
    set_dependent_params(params)
    assert params

    # Create the context.
    ctx = Context(dir_path)
    ctx.params = params
    ctx.probe = probe
    ctx.raw_data = raw_data

    # Load the intermediate results to avoid recomputing things.
    ctx.load()
    ir = ctx.intermediate

    ir.Nbatch = get_Nbatch(raw_data, params)

    # -------------------------------------------------------------------------
    # Find good channels.
    if params.minfr_goodchannels > 0:  # discard channels that have very few spikes
        if 'igood' not in ir:
            # determine bad channels
            ir.igood = get_good_channels(raw_data=raw_data, probe=probe, params=params)
            # Cache the result.
            ctx.write(igood=ir.igood)

        # it's enough to remove bad channels from the channel map, which treats them
        # as if they are dead
        ir.igood = ir.igood.ravel()
        probe.chanMap = probe.chanMap[ir.igood]
        probe.xc = probe.xc[ir.igood]  # removes coordinates of bad channels
        probe.yc = probe.yc[ir.igood]
        probe.kcoords = probe.kcoords[ir.igood]
    probe.Nchan = len(probe.chanMap)  # total number of good channels that we will spike sort

    # upper bound on the number of templates we can have
    params.Nfilt = params.nfilt_factor * probe.Nchan

    # -------------------------------------------------------------------------
    # Find the whitening matrix.
    if 'Wrot' not in ir:
        # outputs a rotation matrix (Nchan by Nchan) which whitens the zero-timelag covariance
        # of the data
        ir.Wrot = get_whitening_matrix(raw_data=raw_data, probe=probe, params=params)
        # Cache the result.
        ctx.write(Wrot=ir.Wrot)

    # -------------------------------------------------------------------------
    # Preprocess data to create proc.dat
    ir.proc_path = dir_path / 'proc.dat'
    if not ir.proc_path.exists():
        # Do not preprocess again if the proc.dat file already exists.
        preprocess(ctx)

    # Open the proc file.
    assert ir.proc_path.exists()
    ir.proc = np.memmap(ir.proc_path, dtype=raw_data.dtype, mode='r', order='F')

    # -------------------------------------------------------------------------
    # Time-reordering as a function of drift.
    #
    # This function saves:
    #
    #       iorig, ccb0, ccbsort
    #
    if 'iorig' not in ir:
        clusterSingleBatches(ctx)

    # -------------------------------------------------------------------------
    # Main tracking and template matching algorithm.
    #
    # This function uses:
    #
    #         proc file
    #         iorig
    #
    # This function saves:
    #
    #         wPCA, wTEMP
    #         st3, simScore
    #         cProj, cProjPC
    #         iNeigh, iNeighPC
    #         WA, UA, W, U, dWU, mu,
    #         W_a, W_b, U_a, U_b
    #
    if 'st3' not in ir:
        learnAndSolve8b(ctx)
    logger.info("%d spikes.", ir.st3.shape[0])

    # -------------------------------------------------------------------------
    # Final merges.
    #
    # This function uses:
    #
    #       st3, simScore
    #
    # This function saves:
    #
    #         R_CCG, Q_CCG, K_CCG
    #
    if 'st3_after_merges' not in ir:
        find_merges(ctx, True)
    logger.info("%d spikes after merge.", ir.st3_after_merges.shape[0])

    # -------------------------------------------------------------------------
    # Final splits.
    #
    # This function uses:
    #
    #       st3_after_merges
    #       wPCA, W, dWU, cProjPC
    #
    # This function saves:
    #
    #       st3_after_split
    #       W, U, mu, Wphy, simScore
    #       iNeigh, iNeighPC, iList, isplit
    #
    if 'st3_after_split' not in ir:
        # final splits by SVD
        splitAllClusters(ctx, True)
        # final splits by amplitudes
        splitAllClusters(ctx, False)
    logger.info("%d spikes after split.", ir.st3_after_split.shape[0])

    # -------------------------------------------------------------------------
    # Decide on cutoff.
    #
    # This function uses:
    #
    #       st3_after_split
    #       wPCA, W, dWU, cProjPC
    #
    # This function saves:
    #
    #       st3_after_cutoff
    #       cProj, cProjPC
    #       est_contam_rate, Ths, good
    #
    if 'st3_after_cutoff' not in ir:
        set_cutoff(ctx)
    logger.info("%d spikes after cutoff.", ir.st3_after_cutoff.shape[0])

    logger.info('Found %d good units.', np.sum(ir.good > 0))

    # write to Phy
    logger.info('Saving results to phy.')
    rezToPhy(ctx, dat_path=dat_path, output_dir=dir_path / 'output')
