import logging
from math import ceil
from functools import lru_cache

import numpy as np
from scipy.signal import butter
import cupy as cp
from tqdm import tqdm

from .cptools import lfilter, _get_lfilter_fun, median, convolve_gpu
from .utils import _make_fortran

logger = logging.getLogger(__name__)


def get_filter_params(fs, fshigh=None, fslow=None):
    if fslow and fslow < fs / 2:
        # butterworth filter with only 3 nodes (otherwise it's unstable for float32)
        return butter(3, (2 * fshigh / fs, 2 * fslow / fs), 'bandpass')
    else:
        # butterworth filter with only 3 nodes (otherwise it's unstable for float32)
        return butter(3, fshigh / fs * 2, 'high')


def gpufilter(buff, chanMap=None, fs=None, fslow=None, fshigh=None, car=True):
    # filter this batch of data after common average referencing with the
    # median
    # buff is timepoints by channels
    # chanMap are indices of the channels to be kep
    # params.fs and params.fshigh are sampling and high-pass frequencies respectively
    # if params.fslow is present, it is used as low-pass frequency (discouraged)

    # set up the parameters of the filter
    b1, a1 = get_filter_params(fs, fshigh=fshigh, fslow=fslow)

    dataRAW = buff  # .T  # NOTE: we no longer use Fortran order upstream
    assert dataRAW.flags.c_contiguous
    assert dataRAW.ndim == 2
    assert dataRAW.shape[0] > dataRAW.shape[1]
    if chanMap is not None and len(chanMap):
        dataRAW = dataRAW[:, chanMap]  # subsample only good channels
    assert dataRAW.ndim == 2

    # subtract the mean from each channel
    dataRAW = dataRAW - cp.mean(dataRAW, axis=0)  # subtract mean of each channel
    assert dataRAW.ndim == 2

    # CAR, common average referencing by median
    if car:
        # subtract median across channels
        dataRAW = dataRAW - median(dataRAW, axis=1)[:, np.newaxis]

    # next four lines should be equivalent to filtfilt (which cannot be
    # used because it requires float64)
    datr = lfilter(b1, a1, dataRAW, axis=0)  # causal forward filter
    datr = lfilter(b1, a1, datr, axis=0, reverse=True)  # backward
    return datr


def _is_vect(x):
    return hasattr(x, '__len__') and len(x) > 1


def _make_vect(x):
    if not hasattr(x, '__len__'):
        x = np.array([x])
    return x


def my_min(S1, sig, varargin=None):
    # returns a running minimum applied sequentially across a choice of dimensions and bin sizes
    # S1 is the matrix to be filtered
    # sig is either a scalar or a sequence of scalars, one for each axis to be filtered.
    #  it's the plus/minus bin length for the minimum filter
    # varargin can be the dimensions to do filtering, if len(sig) != x.shape
    # if sig is scalar and no axes are provided, the default axis is 2
    idims = 1
    if varargin is not None:
        idims = varargin
    idims = _make_vect(idims)
    if _is_vect(idims) and _is_vect(sig):
        sigall = sig
    else:
        sigall = np.tile(sig, len(idims))

    for sig, idim in zip(sigall, idims):
        Nd = S1.ndim
        S1 = cp.transpose(S1, [idim] + list(range(0, idim)) + list(range(idim + 1, Nd)))
        dsnew = S1.shape
        S1 = cp.reshape(S1, (S1.shape[0], -1), order='F' if S1.flags.f_contiguous else 'C')
        dsnew2 = S1.shape
        S1 = cp.concatenate(
            (cp.full((sig, dsnew2[1]), np.inf), S1, cp.full((sig, dsnew2[1]), np.inf)), axis=0)
        Smax = S1[:dsnew2[0], :]
        for j in range(1, 2 * sig + 1):
            Smax = cp.minimum(Smax, S1[j:j + dsnew2[0], :])
        S1 = cp.reshape(Smax, dsnew, order='F' if S1.flags.f_contiguous else 'C')
        S1 = cp.transpose(S1, list(range(1, idim + 1)) + [0] + list(range(idim + 1, Nd)))
    return S1


def my_sum(S1, sig, varargin=None):
    # returns a running sum applied sequentially across a choice of dimensions and bin sizes
    # S1 is the matrix to be filtered
    # sig is either a scalar or a sequence of scalars, one for each axis to be filtered.
    #  it's the plus/minus bin length for the summing filter
    # varargin can be the dimensions to do filtering, if len(sig) != x.shape
    # if sig is scalar and no axes are provided, the default axis is 2
    idims = 1
    if varargin is not None:
        idims = varargin
    idims = _make_vect(idims)
    if _is_vect(idims) and _is_vect(sig):
        sigall = sig
    else:
        sigall = np.tile(sig, len(idims))

    for sig, idim in zip(sigall, idims):
        Nd = S1.ndim
        S1 = cp.transpose(S1, [idim] + list(range(0, idim)) + list(range(idim + 1, Nd)))
        dsnew = S1.shape
        S1 = cp.reshape(S1, (S1.shape[0], -1), order='F')
        dsnew2 = S1.shape
        S1 = cp.concatenate(
            (cp.full((sig, dsnew2[1]), 0), S1, cp.full((sig, dsnew2[1]), 0)), axis=0)
        Smax = S1[:dsnew2[0], :]
        for j in range(1, 2 * sig + 1):
            Smax = Smax + S1[j:j + dsnew2[0], :]
        S1 = cp.reshape(Smax, dsnew, order='F')
        S1 = cp.transpose(S1, list(range(1, idim + 1)) + [0] + list(range(idim + 1, Nd)))
    return S1


def my_conv2(x, sig, varargin=None, **kwargs):
    # x is the matrix to be filtered along a choice of axes
    # sig is either a scalar or a sequence of scalars, one for each axis to be filtered
    # varargin can be the dimensions to do filtering, if len(sig) != x.shape
    # if sig is scalar and no axes are provided, the default axis is 2
    if sig <= .25:
        return x
    idims = 1
    if varargin is not None:
        idims = varargin
    idims = _make_vect(idims)
    if _is_vect(idims) and _is_vect(sig):
        sigall = sig
    else:
        sigall = np.tile(sig, len(idims))

    for sig, idim in zip(sigall, idims):
        Nd = x.ndim
        x = cp.transpose(x, [idim] + list(range(0, idim)) + list(range(idim + 1, Nd)))
        dsnew = x.shape
        x = cp.reshape(x, (x.shape[0], -1), order='F')

        tmax = ceil(4 * sig)
        dt = cp.arange(-tmax, tmax + 1)
        gaus = cp.exp(-dt ** 2 / (2 * sig ** 2))
        gaus = gaus / cp.sum(gaus)

        y = convolve_gpu(x, gaus, **kwargs)
        y = y.reshape(dsnew, order='F')
        y = cp.transpose(y, list(range(1, idim + 1)) + [0] + list(range(idim + 1, Nd)))
    return y


def whiteningFromCovariance(CC):
    # function Wrot = whiteningFromCovariance(CC)
    # takes as input the matrix CC of channel pairwise correlations
    # outputs a symmetric rotation matrix (also Nchan by Nchan) that rotates
    # the data onto uncorrelated, unit-norm axes

    # covariance eigendecomposition (same as svd for positive-definite matrix)
    E, D, _ = cp.linalg.svd(CC)
    eps = 1e-6
    Di = cp.diag(1. / (D + eps) ** .5)
    Wrot = cp.dot(cp.dot(E, Di), E.T)  # this is the symmetric whitening matrix (ZCA transform)
    return Wrot


def whiteningLocal(CC, yc, xc, nRange):
    # function to perform local whitening of channels
    # CC is a matrix of Nchan by Nchan correlations
    # yc and xc are vector of Y and X positions of each channel
    # nRange is the number of nearest channels to consider
    Wrot = cp.zeros((CC.shape[0], CC.shape[0]))

    for j in range(CC.shape[0]):
        ds = (xc - xc[j]) ** 2 + (yc - yc[j]) ** 2
        ilocal = np.argsort(ds)
        # take the closest channels to the primary channel.
        # First channel in this list will always be the primary channel.
        ilocal = ilocal[:nRange]

        wrot0 = cp.asnumpy(whiteningFromCovariance(CC[np.ix_(ilocal, ilocal)]))
        # the first column of wrot0 is the whitening filter for the primary channel
        Wrot[ilocal, j] = wrot0[:, 0]

    return Wrot


def get_whitening_matrix(raw_data=None, probe=None, params=None):
    """
    based on a subset of the data, compute a channel whitening matrix
    this requires temporal filtering first (gpufilter)
    """
    Nbatch = get_Nbatch(raw_data, params)
    ntbuff = params.ntbuff
    NTbuff = params.NTbuff
    whiteningRange = params.whiteningRange
    scaleproc = params.scaleproc
    NT = params.NT
    fs = params.fs
    fshigh = params.fshigh
    nSkipCov = params.nSkipCov

    xc = probe.xc
    yc = probe.yc
    chanMap = probe.chanMap
    Nchan = probe.Nchan
    chanMap = probe.chanMap

    # Nchan is obtained after the bad channels have been removed
    CC = cp.zeros((Nchan, Nchan))

    for ibatch in tqdm(range(0, Nbatch, nSkipCov), desc="Computing the whitening matrix"):
        i = max(0, (NT - ntbuff) * ibatch - 2 * ntbuff)
        # WARNING: we no longer use Fortran order, so raw_data is nsamples x NchanTOT
        buff = raw_data[i:i + NT - ntbuff]
        assert buff.shape[0] > buff.shape[1]
        assert buff.flags.c_contiguous

        nsampcurr = buff.shape[0]
        if nsampcurr < NTbuff:
            buff = np.concatenate(
                (buff, np.tile(buff[nsampcurr - 1], (NTbuff, 1))), axis=0)

        buff_g = cp.asarray(buff, dtype=np.float32)

        # apply filters and median subtraction
        datr = gpufilter(buff_g, fs=fs, fshigh=fshigh, chanMap=chanMap)
        assert datr.flags.c_contiguous

        CC = CC + cp.dot(datr.T, datr) / NT  # sample covariance

    CC = CC / ceil((Nbatch - 1) / nSkipCov)

    if whiteningRange < np.inf:
        #  if there are too many channels, a finite whiteningRange is more robust to noise
        # in the estimation of the covariance
        whiteningRange = min(whiteningRange, Nchan)
        # this function performs the same matrix inversions as below, just on subsets of
        # channels around each channel
        Wrot = whiteningLocal(CC, yc, xc, whiteningRange)
    else:
        Wrot = whiteningFromCovariance(CC)

    Wrot = Wrot * scaleproc

    logger.info("Computed the whitening matrix.")

    return Wrot


def get_good_channels(raw_data=None, probe=None, params=None):
    """
    of the channels indicated by the user as good (chanMap)
    further subset those that have a mean firing rate above a certain value
    (default is ops.minfr_goodchannels = 0.1Hz)
    needs the same filtering parameters in ops as usual
    also needs to know where to start processing batches (twind)
    and how many channels there are in total (NchanTOT)
    """
    fs = params.fs
    fshigh = params.fshigh
    fslow = params.fslow
    Nbatch = get_Nbatch(raw_data, params)
    NT = params.NT
    spkTh = params.spkTh
    nt0 = params.nt0
    minfr_goodchannels = params.minfr_goodchannels

    chanMap = probe.chanMap
    # Nchan = probe.Nchan
    NchanTOT = len(chanMap)

    ich = []
    k = 0
    ttime = 0

    # skip every 100 batches
    for ibatch in tqdm(range(0, Nbatch, int(ceil(Nbatch / 100))), desc="Finding good channels"):
        i = NT * ibatch
        buff = raw_data[i:i + NT]
        # buff = _make_fortran(buff)
        # NOTE: using C order now
        assert buff.shape[0] > buff.shape[1]
        assert buff.flags.c_contiguous
        if buff.size == 0:
            break

        # Put on GPU.
        buff = cp.asarray(buff, dtype=np.float32)
        assert buff.flags.c_contiguous
        datr = gpufilter(buff, chanMap=chanMap, fs=fs, fshigh=fshigh, fslow=fslow)
        assert datr.shape[0] > datr.shape[1]

        # very basic threshold crossings calculation
        s = cp.std(datr, axis=0)
        datr = datr / s  # standardize each channel ( but don't whiten)
        mdat = my_min(datr, 30, 0)  # get local minima as min value in +/- 30-sample range

        # take local minima that cross the negative threshold
        xi, xj = cp.nonzero((datr < mdat + 1e-3) & (datr < spkTh))

        # filtering may create transients at beginning or end. Remove those.
        xj = xj[(xi >= nt0) & (xi <= NT - nt0)]

        # collect the channel identities for the detected spikes
        ich.append(xj)
        k += xj.size

        # keep track of total time where we took spikes from
        ttime += datr.shape[0] / fs

    ich = cp.concatenate(ich)

    # count how many spikes each channel got
    nc, _ = cp.histogram(ich, cp.arange(NchanTOT + 1))

    # divide by total time to get firing rate
    nc = nc / ttime

    # keep only those channels above the preset mean firing rate
    igood = cp.asnumpy(nc >= minfr_goodchannels)

    if len(igood) == 0:
        raise RuntimeError("No good channels found! Verify your raw data and parameters.")

    logger.info('Found %d threshold crossings in %2.2f seconds of data.' % (k, ttime))
    logger.info('Found %d/%d bad channels.' % (np.sum(~igood), len(igood)))

    return igood


def get_Nbatch(raw_data, params):
    n_samples = max(raw_data.shape)
    # we assume raw_data as been already virtually split with the requested trange
    return ceil(n_samples / (params.NT - params.ntbuff))  # number of data batches


def preprocess(ctx):
    # function rez = preprocessDataSub(ops)
    # this function takes an ops struct, which contains all the Kilosort2 settings and file paths
    # and creates a new binary file of preprocessed data, logging new variables into rez.
    # The following steps are applied:
    # 1) conversion to float32
    # 2) common median subtraction
    # 3) bandpass filtering
    # 4) channel whitening
    # 5) scaling to int16 values

    params = ctx.params
    probe = ctx.probe
    raw_data = ctx.raw_data
    ir = ctx.intermediate

    fs = params.fs
    fshigh = params.fshigh
    fslow = params.fslow
    Nbatch = ir.Nbatch
    NT = params.NT
    NTbuff = params.NTbuff

    Wrot = cp.asarray(ir.Wrot)

    logger.info("Loading raw data and applying filters.")

    with open(ir.proc_path, 'wb') as fw:  # open for writing processed data
        for ibatch in tqdm(range(Nbatch), desc="Preprocessing"):
            # we'll create a binary file of batches of NT samples, which overlap consecutively
            # on params.ntbuff samples
            # in addition to that, we'll read another params.ntbuff samples from before and after,
            # to have as buffers for filtering

            # number of samples to start reading at.
            i = max(0, (NT - params.ntbuff) * ibatch - 2 * params.ntbuff)
            if ibatch == 0:
                # The very first batch has no pre-buffer, and has to be treated separately
                ioffset = 0
            else:
                ioffset = params.ntbuff

            buff = raw_data[i:i + NTbuff]
            if buff.size == 0:
                logger.error("Loaded buffer has an empty size!")
                break  # this shouldn't really happen, unless we counted data batches wrong

            nsampcurr = buff.shape[0]  # how many time samples the current batch has
            if nsampcurr < NTbuff:
                buff = np.concatenate(
                    (buff, np.tile(buff[nsampcurr - 1], (NTbuff, 1))), axis=0)

            # apply filters and median subtraction
            buff = cp.asarray(buff, dtype=np.float32)

            datr = gpufilter(buff, chanMap=probe.chanMap, fs=fs, fshigh=fshigh, fslow=fslow)
            assert datr.flags.c_contiguous

            datr = datr[ioffset:ioffset + NT, :]  # remove timepoints used as buffers
            datr = cp.dot(datr, Wrot)  # whiten the data and scale by 200 for int16 range
            assert datr.flags.c_contiguous

            # convert to int16, and gather on the CPU side
            # WARNING: transpose because "tofile" always writes in C order, whereas we want
            # to write in F order.
            datcpu = cp.asnumpy(datr.T.astype(np.int16))

            # write this batch to binary file
            datcpu.tofile(fw)
