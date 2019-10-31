import logging
from math import sqrt, ceil

import numpy as np
import cupy as cp
from tqdm import tqdm

from .cptools import svdecon, svdecon_cpu, median, free_gpu_memory, ones
from .cluster import isolated_peaks_new, get_SpikeSample, getClosestChannels
from .utils import get_cuda

logger = logging.getLogger(__name__)


def extractTemplatesfromSnippets(proc=None, probe=None, params=None, Nbatch=None, nPCs=None):
    # this function is very similar to extractPCfromSnippets.
    # outputs not just the PC waveforms, but also the template "prototype",
    # basically k-means clustering of 1D waveforms.

    NT = params.NT
    # skip every this many batches
    nskip = params.nskip
    nPCs = nPCs or params.nPCs
    nt0min = params.nt0min
    Nchan = probe.Nchan
    batchstart = np.arange(0, NT * Nbatch + 1, NT)

    k = 0
    # preallocate matrix to hold 1D spike snippets
    dd = cp.zeros((params.nt0, int(5e4)), dtype=np.float32, order='F')

    for ibatch in tqdm(range(0, Nbatch, nskip), desc="Extracting templates"):
        offset = Nchan * batchstart[ibatch]
        dat = proc.flat[offset:offset + NT * Nchan].reshape((-1, Nchan), order='F')

        # move data to GPU and scale it back to unit variance
        dataRAW = cp.asarray(dat, dtype=np.float32) / params.scaleproc

        # find isolated spikes from each batch
        row, col, mu = isolated_peaks_new(dataRAW, params)

        # for each peak, get the voltage snippet from that channel
        c = get_SpikeSample(dataRAW, row, col, params)

        if k + c.shape[1] > dd.shape[1]:
            dd[:, 2 * dd.shape[1]] = 0

        dd[:, k:k + c.shape[1]] = c
        k = k + c.shape[1]
        if k > 1e5:
            break

    # discard empty samples
    dd = dd[:, :k]

    # initialize the template clustering with random waveforms
    uu = np.random.permutation(dd.shape[1])[:nPCs]
    wTEMP = dd[:, uu]
    wTEMP = wTEMP / cp.sum(wTEMP ** 2, axis=0) ** .5  # normalize them

    for i in range(10):
        # at each iteration, assign the waveform to its most correlated cluster
        cc = cp.dot(wTEMP.T, dd)
        imax = cp.argmax(cc, axis=0)
        amax = cc[imax, np.arange(cc.shape[1])]
        for j in range(nPCs):
            # weighted average to get new cluster means
            wTEMP[:, j] = cp.dot(dd[:, imax == j], amax[imax == j].T)
        wTEMP = wTEMP / cp.sum(wTEMP ** 2, axis=0) ** .5  # unit normalize

    # the PCs are just the left singular vectors of the waveforms
    U, Sv, V = svdecon(dd)

    # take as many as needed
    wPCA = U[:, :nPCs]

    # adjust the arbitrary sign of the first PC so its negativity is downward
    wPCA[:, 0] = -wPCA[:, 0] * cp.sign(wPCA[nt0min, 0])

    return wTEMP, wPCA


def getKernels(params):
    # this function makes upsampling kernels for the temporal components.
    # those are used for interpolating the biggest negative peak,
    # and aligning the template to that peak with sub-sample resolution
    # needs nup, the interpolation factor (default = 10)
    # also needs sig, the interpolation smoothness (default = 1)

    nup = params.nup
    sig = params.sig

    nt0min = params.nt0min
    nt0 = params.nt0

    xs = cp.arange(1, nt0 + 1)
    ys = cp.linspace(.5, nt0 + .5, nt0 * nup + 1)[:-1]

    # these kernels are just standard kriging interpolators

    # first compute distances between the sample coordinates
    # for some reason, this seems to be circular, although the waveforms are not circular
    # I think the reason had to do with some constant offsets in some channels?
    d = cp.mod(xs[:, np.newaxis] - xs[np.newaxis, :] + nt0, nt0)
    d = cp.minimum(d, nt0 - d)
    # the kernel covariance uses a squared exponential of spatial scale sig
    Kxx = cp.exp(-d ** 2 / sig ** 2)

    # do the same for the kernel similarities between upsampled "test" timepoints and
    # the original coordinates
    d = cp.mod(ys[:, np.newaxis] - xs[np.newaxis, :] + nt0, nt0)
    d = cp.minimum(d, nt0 - d)
    Kyx = cp.exp(-d ** 2 / sig ** 2)

    # the upsampling matrix is given by the following formula,
    # with some light diagonal regularization of the matrix inversion
    B = cp.dot(Kyx, cp.linalg.inv(Kxx + .01 * cp.eye(nt0)))
    B = B.reshape((nup, nt0, nt0), order='F')

    # A is just a slice through this upsampling matrix corresponding to the most negative point
    # this is used to compute the biggest negative deflection (after upsampling)
    A = cp.squeeze(B[:, nt0min - 1, :])
    B = cp.transpose(B, [1, 2, 0])

    return A.astype(np.float64), B.astype(np.float64)


def memorizeW(W, U, mu):
    Wraw = cp.zeros((U.shape[0], W.shape[0], U.shape[1]), dtype=np.float64, order='F')

    for n in range(U.shape[1]):
        # temporarily use U rather Urot until I have a chance to test it
        Wraw[:, :, n] = mu[n] * cp.dot(U[:, n, :], W[:, n, :].T)

    return Wraw


def getMeUtU(iU, iC, mask, Nnearest, Nchan):
    # function [UtU, maskU, iList] = getMeUtU(iU, iC, mask, Nnearest, Nchan)
    # this function determines if two templates share any channels
    # iU are the channels that each template is assigned to, one main channel per template
    # iC has as column K the list of neigboring channels for channel K
    # mask are the weights assigned for the corresponding neighboring channels
    # in iC (gaussian-decaying)

    Nfilt = iU.size

    # create a sparse matrix with ones if a channel K belongs to a template
    U = cp.zeros((Nchan, Nfilt), dtype=np.float32, order='F')

    # use the template primary channel to obtain its neighboring channels from iC
    ix = iC[:, iU] + cp.arange(0, Nchan * Nfilt, Nchan).astype(np.int32)
    U[ix] = 1  # use this as an awkward index into U

    # if this is 0, the templates had not pair of channels in common
    UtU = (cp.dot(U.T, U) > 0).astype(np.int32)

    # we also return the masks for each template, picked from the corresponding mask of
    # their primary channel
    maskU = mask[:, iU]

    # sort template pairs in order of how many channels they share
    isort = cp.argsort(UtU, axis=0)[::-1]
    iList = isort[:Nnearest, :]  # take the Nnearest templates for each template

    return UtU, maskU, iList


def getMeWtW2(W, U0, Nnearest=None):
    # this function compute the correlation between any two pairs of templates
    # it relies on the fact that the W and U0 are unit normalized, so that the product of a
    # template with itself is 1, as it should be if we're trying to calculate correlations

    # takes input the temporal and spatial factors of the low-rank template, as
    # well as the number of most similar template pairs desired to be output in
    # iList

    nt0, Nfilt, Nrank = W.shape
    WtW = cp.zeros((Nfilt, Nfilt), dtype=np.float32, order='F')

    # since the templates are factorized into orthonormal components, we can compute dot products
    # one dimension at a time
    for i in range(Nrank):
        for j in range(Nrank):
            #  this computes the spatial dot product
            utu0 = cp.dot(U0[:, :, i].T, U0[:, :, j])
            #  this computes the temporal dot product
            wtw0 = cp.dot(W[:, :, i].T, W[:, :, j])

            # the element-wise product of these is added to the matrix of correlatioons
            WtW = WtW + wtw0 * utu0

    # also return a list of most correlated template pairs
    isort = cp.argsort(WtW, axis=0)[::-1]

    if Nnearest:
        # if we don't have enough templates yet, just wrap the indices around the range 1:Nfilt
        iNear = cp.mod(cp.arange(Nnearest), Nfilt)
        iList = isort[iNear, :]  # return the list of pairs for each template
        return WtW, iList
    else:
        return WtW


def mexGetSpikes2(Params, drez, wTEMP, iC):
    code, constants = get_cuda('mexGetSpikes2')

    NT = int(Params[0])
    Nchan = int(Params[9])
    nt0 = int(Params[4])
    # Nnearest = int(Params[5])
    Nrank = int(Params[14])

    maxFR = constants.maxFR
    Nthreads = constants.Nthreads

    # tpB = (8, 2 * nt0 - 1)
    # tpF = (16, Nnearest)
    tpS = (nt0, 16)

    d_Params = cp.asarray(Params, dtype=np.float64, order='F')
    d_data = cp.asarray(drez, dtype=np.float32, order='F')
    d_W = cp.asarray(wTEMP, dtype=np.float32, order='F')
    d_iC = cp.asarray(iC, dtype=np.int32, order='F')

    d_counter = cp.zeros(2, dtype=np.int32, order='F')
    d_dout = cp.zeros((NT, Nchan), dtype=np.float32, order='F')
    d_dfilt = cp.zeros((Nrank, NT, Nchan), dtype=np.float32, order='F')
    d_err = cp.zeros(NT, dtype=np.float32, order='F')
    d_kkmax = cp.zeros((NT, Nchan), dtype=np.int32, order='F')
    d_kk = cp.zeros(NT, dtype=np.int32, order='F')
    d_ftype = cp.zeros(NT, dtype=np.int32, order='F')
    d_st = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_id = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_x = cp.zeros(maxFR, dtype=np.float32, order='F')
    d_st1 = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_id1 = cp.zeros(maxFR, dtype=np.int32, order='F')

    counter = np.zeros(2, dtype=np.int32, order='F')

    # filter the data with the temporal templates
    Conv1D = cp.RawKernel(code, 'Conv1D')
    Conv1D((Nchan,), (Nthreads,), (d_Params, d_data, d_W, d_dfilt))

    # sum each template across channels, square, take max
    sumChannels = cp.RawKernel(code, 'sumChannels')
    sumChannels((int(NT / Nthreads),), (Nthreads,), (d_Params, d_dfilt, d_dout, d_kkmax, d_iC))

    # compute the best filter
    bestFilter = cp.RawKernel(code, 'bestFilter')
    bestFilter(
        (int(NT / Nthreads),), (Nthreads,), (d_Params, d_dout, d_err, d_ftype, d_kkmax, d_kk))

    # ignore peaks that are smaller than another nearby peak
    cleanup_spikes = cp.RawKernel(code, 'cleanup_spikes')
    cleanup_spikes(
        (int(NT / Nthreads),), (Nthreads,), (d_Params, d_err, d_ftype, d_x, d_st, d_id, d_counter))

    # ignore peaks that are smaller than another nearby peak
    cleanup_heights = cp.RawKernel(code, 'cleanup_heights')
    cleanup_heights(
        (1 + int(maxFR // 32),), (32,), (d_Params, d_x, d_st, d_id, d_st1, d_id1, d_counter))

    # add new spikes to 2nd counter
    counter[0] = d_counter[1]
    counter[0] = min(maxFR, counter[0])

    d_WU = cp.zeros((nt0, Nchan, counter[0]), dtype=np.float32, order='F')
    # d_WU1 = cp.zeros((nt0, Nchan, counter[0]), dtype=np.float32, order='F')

    # update dWU here by adding back to subbed spikes
    extract_snips = cp.RawKernel(code, 'extract_snips')
    extract_snips((Nchan,), tpS, (d_Params, d_st1, d_id1, d_counter, d_data, d_WU))

    # QUESTION: why a copy here??
    # if counter[0] > 0:
    #     d_WU1[...] = d_WU[...]

    del (
        d_ftype, d_kkmax, d_err, d_st, d_id, d_st1, d_x, d_kk, d_id1, d_counter,
        d_Params, d_dfilt)
    return d_WU, d_dout


def mexSVDsmall2(Params, dWU, W, iC, iW, Ka, Kb):
    code, constants = get_cuda('mexSVDsmall2')

    Nthreads = constants.Nthreads

    Nfilt = int(Params[1])
    nt0 = int(Params[4])
    Nrank = int(Params[6])
    Nchan = int(Params[9])

    # print("Nfilt", Nfilt)
    # print("nt0", nt0)
    # print("Nrank", Nrank)
    # print("Nchan", Nchan)
    # print("dWU", dWU.shape)
    # print("W", W.shape)
    # print("iC", iC.shape)
    # print("iW", iW.shape)
    # print("Ka", Ka.shape)
    # print("Kb", Kb.shape)

    d_Params = cp.asarray(Params, dtype=np.float64, order='F')

    d_dWU = cp.asarray(dWU, dtype=np.float64, order='F')
    d_iC = cp.asarray(iC, dtype=np.int32, order='F')
    d_iW = cp.asarray(iW, dtype=np.int32, order='F')

    d_A = cp.asarray(Ka, dtype=np.float64, order='F')
    d_B = cp.asarray(Kb, dtype=np.float64, order='F')

    d_U = cp.zeros((Nchan, Nfilt, Nrank), dtype=np.float64, order='F')
    d_mu = cp.zeros(Nfilt, dtype=np.float64, order='F')

    d_W = cp.asarray(W, dtype=np.float64, order='F')

    d_wtw = cp.zeros((nt0, nt0, Nfilt), dtype=np.float64, order='F')
    d_dWUb = cp.zeros((nt0, Nchan, Nfilt), dtype=np.float64, order='F')

    tpS = (nt0, int(Nthreads // nt0))
    tpK = (Nrank, int(Nthreads // Nrank))

    blankdWU = cp.RawKernel(code, 'blankdWU')
    blankdWU((Nfilt,), tpS, (d_Params, d_dWU, d_iC, d_iW, d_dWUb))

    # compute dWU * dWU'
    getwtw = cp.RawKernel(code, 'getwtw')
    getwtw((Nfilt,), tpS, (d_Params, d_dWUb, d_wtw))

    # get W by power svd iterations
    getW = cp.RawKernel(code, 'getW')
    getW((Nfilt,), (nt0,), (d_Params, d_wtw, d_W))

    # compute U by W' * dWU
    getU = cp.RawKernel(code, 'getU')
    getU((Nfilt,), tpK, (d_Params, d_dWUb, d_W, d_U))

    # normalize U, get S, get mu, renormalize W
    reNormalize = cp.RawKernel(code, 'reNormalize')
    reNormalize((Nfilt,), (nt0,), (d_Params, d_A, d_B, d_W, d_U, d_mu))

    del d_wtw, d_Params, d_dWUb

    return d_W, d_U, d_mu


def mexMPnu8(Params, dataRAW, U, W, mu, iC, iW, UtU, iList, wPCA):
    code, constants = get_cuda('mexMPnu8')
    maxFR = int(constants.maxFR)
    nmaxiter = int(constants.nmaxiter)
    Nthreads = int(constants.Nthreads)

    NT = int(Params[0])
    Nfilt = int(Params[1])
    nt0 = int(Params[4])
    Nnearest = int(Params[5])
    Nrank = int(Params[6])
    NchanU = int(Params[10])
    Nchan = int(Params[9])

    d_Params = cp.asarray(Params, dtype=np.float64, order='F')

    d_draw = cp.asarray(dataRAW, dtype=np.float32, order='F')
    d_U = cp.asarray(U, dtype=np.float32, order='F')
    d_W = cp.asarray(W, dtype=np.float32, order='F')
    d_mu = cp.asarray(mu, dtype=np.float32, order='F')
    d_iC = cp.asarray(iC, dtype=np.int32, order='F')
    d_iW = cp.asarray(iW, dtype=np.int32, order='F')
    d_UtU = cp.asarray(UtU, dtype=np.bool, order='F')
    d_iList = cp.asarray(iList, dtype=np.int32, order='F')
    d_wPCA = cp.asarray(wPCA, dtype=np.float32, order='F')

    d_nsp = cp.zeros(Nfilt, dtype=np.int32, order='F')
    d_dWU = cp.zeros((nt0, Nchan, Nfilt), dtype=np.float64, order='F')

    d_dout = cp.zeros((2 * NT, Nfilt), dtype=np.float32, order='F')
    d_data = cp.zeros((NT, Nfilt, Nrank), dtype=np.float32, order='F')
    d_err = cp.zeros(NT, dtype=np.float32, order='F')
    d_ftype = cp.zeros(NT, dtype=np.int32, order='F')
    d_eloss = cp.zeros(NT, dtype=np.float32, order='F')
    d_st = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_id = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_x = cp.zeros(maxFR, dtype=np.float32, order='F')
    d_y = cp.zeros(maxFR, dtype=np.float32, order='F')
    d_z = cp.zeros(maxFR, dtype=np.float32, order='F')

    d_counter = cp.zeros(2, dtype=np.int32, order='F')
    d_count = cp.zeros(nmaxiter, dtype=np.int32, order='F')
    d_feat = cp.zeros((Nnearest, maxFR), dtype=np.float32, order='F')
    d_featPC = cp.zeros((NchanU, Nrank, maxFR), dtype=np.float32, order='F')

    counter = np.zeros(2, dtype=np.int32, order='F')

    # tpB = (8, 2 * nt0 - 1)
    tpF = (16, Nnearest)
    tpS = (nt0, 16)
    # tpW = (Nnearest, Nrank)
    tpPC = (NchanU, Nrank)

    # filter the data with the spatial templates
    spaceFilter = cp.RawKernel(code, 'spaceFilter')
    spaceFilter((Nfilt,), (Nthreads,), (d_Params, d_draw, d_U, d_iC, d_iW, d_data))

    # filter the data with the temporal templates
    timeFilter = cp.RawKernel(code, 'timeFilter')
    timeFilter((Nfilt,), (Nthreads,), (d_Params, d_data, d_W, d_dout))

    # compute the best filter
    bestFilter = cp.RawKernel(code, 'bestFilter')
    bestFilter(
        (int(NT // Nthreads),), (Nthreads,), (d_Params, d_dout, d_mu, d_err, d_eloss, d_ftype))

    # loop to find and subtract spikes
    for k in range(int(Params[3])):
        # ignore peaks that are smaller than another nearby peak
        cleanup_spikes = cp.RawKernel(code, 'cleanup_spikes')
        cleanup_spikes(
            (int(NT // Nthreads),), (Nthreads,),
            (d_Params, d_dout, d_mu, d_err, d_eloss,
             d_ftype, d_st, d_id, d_x, d_y, d_z, d_counter))

        # add new spikes to 2nd counter
        counter[:] = cp.asnumpy(d_counter[:])
        if counter[0] > maxFR:
            counter[0] = maxFR
            d_counter[0] = counter[0]

        # extract template features before subtraction
        if Params[12] > 1:
            extractFEAT = cp.RawKernel(code, 'extractFEAT')
            extractFEAT(
                (64,), tpF, (d_Params, d_st, d_id, d_counter, d_dout, d_iList, d_mu, d_feat))

        # subtract spikes from raw data here
        subtract_spikes = cp.RawKernel(code, 'subtract_spikes')
        subtract_spikes((Nfilt,), tpS, (d_Params, d_st, d_id, d_y, d_counter, d_draw, d_W, d_U))

        # filter the data with the spatial templates
        spaceFilterUpdate = cp.RawKernel(code, 'spaceFilterUpdate')
        spaceFilterUpdate(
            (Nfilt,), (2 * nt0 - 1,),
            (d_Params, d_draw, d_U, d_UtU, d_iC, d_iW, d_data, d_st, d_id, d_counter))

        # filter the data with the temporal templates
        timeFilterUpdate = cp.RawKernel(code, 'timeFilterUpdate')
        timeFilterUpdate(
            (Nfilt,), (2 * nt0 - 1,),
            (d_Params, d_data, d_W, d_UtU, d_dout, d_st, d_id, d_counter))

        if counter[0] - counter[1] > 0:
            bestFilterUpdate = cp.RawKernel(code, 'bestFilterUpdate')
            bestFilterUpdate(
                (counter[0] - counter[1],), (2 * nt0 - 1,),
                (d_Params, d_dout, d_mu, d_err, d_eloss, d_ftype, d_st, d_id, d_counter))

        d_count[k + 1] = d_counter[0]

        # update 1st counter from 2nd counter
        d_counter[1] = d_counter[0]

    # compute PC features from reziduals + subtractions
    if Params[12] > 0:
        computePCfeatures = cp.RawKernel(code, 'computePCfeatures')
        computePCfeatures(
            (Nfilt,), tpPC,
            (d_Params, d_counter, d_draw, d_st, d_id, d_y,
             d_W, d_U, d_mu, d_iW, d_iC, d_wPCA, d_featPC))

    # update dWU here by adding back to subbed spikes.
    # additional parameter d_idx = array of time sorted indicies
    average_snips = cp.RawKernel(code, 'average_snips')
    average_snips(
        (Nfilt,), tpS,
        (d_Params, d_st, d_id, d_x, d_y, d_counter, d_draw, d_W, d_U, d_dWU, d_nsp, d_mu, d_z))

    if counter[0] < maxFR:
        minSize = counter[0]
    else:
        minSize = maxFR

    del d_counter, d_Params, d_ftype, d_err, d_eloss, d_z, d_dout, d_data

    return (
        d_st[:minSize], d_id[:minSize], d_y[:minSize], d_feat[..., :minSize],
        d_dWU, d_draw, d_nsp, d_featPC[..., :minSize], d_x[:minSize])


def mexWtW2(Params, W1, W2, UtU):
    code, constants = get_cuda('mexWtW2')

    nblock = constants.nblock

    Nfilt = int(Params[1])
    nt0 = int(Params[9])

    d_Params = cp.asarray(Params, dtype=np.float64, order='F')

    d_W1 = cp.asarray(W1, dtype=np.float32, order='F')
    d_W2 = cp.asarray(W2, dtype=np.float32, order='F')
    d_UtU = cp.asarray(UtU, dtype=np.float32, order='F')

    d_WtW = cp.zeros((Nfilt, Nfilt, 2 * nt0 - 1), dtype=np.float32, order='F')

    grid = (1 + int(Nfilt // nblock), 1 + int(Nfilt // nblock))
    block = (nblock, nblock)

    crossFilter = cp.RawKernel(code, 'crossFilter')
    crossFilter(grid, block, (d_Params, d_W1, d_W2, d_UtU, d_WtW))

    del d_Params, d_W1, d_W2, d_UtU

    return d_WtW


def getMeWtW(W, U0, Nnearest=None):
    # this function computes correlation between templates at ALL timelags from each other
    # takes the max over timelags to obtain a similarity score
    # also returns lists of most similar templates to each template
    # takes as input the low-rank factorization of templates (W for time and U0
    # for space)

    # W is timesamples (default = 61 ), by number of templates, by rank (default = 3)
    nt0, Nfilt, Nrank = W.shape

    Params = [1, Nfilt, 0, 0, 0, 0, 0, 0, 0, nt0]

    # initialize correlation matrix for all timelags
    WtW = cp.zeros((Nfilt, Nfilt, 2 * nt0 - 1), dtype=np.float32, order='F')
    for i in range(Nrank):
        for j in range(Nrank):
            # the dot product factorizes into separable products for each spatio-temporal component
            utu0 = cp.dot(U0[:, :, i].T, U0[:, :, j])  # spatial products
            # temporal convolutions get multiplied wit hthe spatial products
            wtw0 = mexWtW2(Params, W[:, :, i], W[:, :, j], utu0)
            # add it to the full correlation array
            WtW = WtW + wtw0

    # the maximum across timelags accounts for sample alignment mismatch
    cc = cp.max(WtW, axis=2)

    if Nnearest:
        isort = cp.argsort(cc, axis=0)[::-1]
        # if we don't have enough templates yet, just wrap the indices around the range 1:Nfilt
        iNear = cp.mod(cp.arange(Nnearest), Nfilt)
        iList = isort[iNear, :]  # return the list of pairs for each template
        return WtW, iList
    else:
        return WtW


def triageTemplates2(params, iW, C2C, W, U, dWU, mu, nsp, ndrop):

    # This function checks if some templates should be dropped
    # either because they are very similar to another template,
    # or because they are not catching any spikes, (low mean firing rate).
    # Takes as inputs almost all the information that determines templates, and
    # outputs the same variables back after removing some clusters.

    # this is the firing rate threshold
    m0 = params.minFR * params.NT / params.fs
    idrop = nsp < m0  # drop any templates with firing rate below this

    # remove those templates everywhere
    W = W[:, ~idrop, :]
    U = U[:, ~idrop, :]
    dWU = dWU[:, :, ~idrop]
    mu = mu[~idrop]
    nsp = nsp[~idrop]
    # keep track of how many templates have been removed this way
    ndrop[0] = .9 * ndrop[0] + .1 * idrop.sum()

    # compute pairwise correlations between templates
    cc = getMeWtW2(W, U, None)
    cc = cc - cp.diag(cp.diag(cc))  # exclude the diagonal

    sd = sqrt(10)  # this is hard-coded here

    # compute a score for the separation of the means
    r0 = 4 * sd / cp.abs(mu[:, np.newaxis] - mu[np.newaxis, :])
    # determine which template has more spikes (that one survives)
    rdir = (nsp[:, np.newaxis] - nsp[np.newaxis, :]) < 0
    # for each pair of template, score their similarity by their template correlation,
    # and amplitude separation
    ipair = (cc > 0.9) & (r0 > 1) & rdir
    # for each template, find its most similar other template
    amax = cp.max(ipair, axis=1)
    # if this score is 1, then all the criteria have bene met for dropping this template
    idrop = amax > 0

    # remove these templates everywhere like before
    W = W[:, ~idrop, :]
    U = U[:, ~idrop, :]
    dWU = dWU[:, :, ~idrop]
    mu = mu[~idrop]
    nsp = nsp[~idrop]
    # keep track of how many templates have been removed this way
    ndrop[1] = .9 * ndrop[1] + .1 * idrop.sum()

    return W, U, dWU, mu, nsp, ndrop


def learnAndSolve8b(ctx):
    """This is the main optimization. Takes the longest time and uses the GPU heavily."""

    Nbatch = ctx.intermediate.Nbatch
    params = ctx.params
    probe = ctx.probe
    ir = ctx.intermediate
    proc = ir.proc

    iorig = ir.iorig

    NrankPC = 6  # this one is the rank of the PCs, used to detect spikes with threshold crossings
    Nrank = 3  # this one is the rank of the templates

    wTEMP, wPCA = extractTemplatesfromSnippets(
        proc=proc, probe=probe, params=params, Nbatch=Nbatch, nPCs=NrankPC)

    # move these to the GPU
    wPCA = cp.asarray(wPCA[:, :Nrank], dtype=np.float32, order='F')
    wTEMP = cp.asarray(wTEMP, dtype=np.float32, order='F')
    wPCAd = cp.asarray(wPCA, dtype=np.float64, order='F')  # convert to double for extra precision

    if 'wPCA' not in ir:
        ctx.save(wPCA=wPCA, wTEMP=wTEMP)

    nt0 = params.nt0
    nt0min = params.nt0min
    nBatches = Nbatch
    NT = params.NT
    Nfilt = params.Nfilt
    Nchan = probe.Nchan

    # two variables for the same thing? number of nearest channels to each primary channel
    NchanNear = min(probe.Nchan, 32)
    Nnearest = min(probe.Nchan, 32)

    # decay of gaussian spatial mask centered on a channel
    sigmaMask = params.sigmaMask

    batchstart = list(range(0, NT * nBatches + 1, NT))

    # find the closest NchanNear channels, and the masks for those channels
    iC, mask, C2C = getClosestChannels(probe, sigmaMask, NchanNear)

    # sorting order for the batches
    isortbatches = iorig
    nhalf = int(ceil(nBatches / 2)) - 1  # halfway point

    # this batch order schedule goes through half of the data forward and backward during the model
    # fitting and then goes through the data symmetrically-out from the center during the final
    # pass
    ischedule = np.concatenate(
        (np.arange(nhalf, nBatches), np.arange(nBatches - 1, nhalf - 1, -1)))
    i1 = np.arange(nhalf - 1, -1, -1)
    i2 = np.arange(nhalf, nBatches)

    irounds = np.concatenate((ischedule, i1, i2))

    niter = irounds.size
    if irounds[niter - nBatches - 1] != nhalf:
        # this check is in here in case I do somehting weird when I try different schedules
        raise ValueError('Mismatch between number of batches')

    # these two flags are used to keep track of what stage of model fitting we're at
    # flag_final = 0
    flag_resort = 1

    # this is the absolute temporal offset in seconds corresponding to the start of the
    # spike sorted time segment
    t0 = 0  # ceil(params.trange(1) * ops.fs)

    nInnerIter = 60  # this is for SVD for the power iteration

    # schedule of learning rates for the model fitting part
    # starts small and goes high, it corresponds approximately to the number of spikes
    # from the past that were averaged to give rise to the current template
    pmi = cp.exp(-1. / cp.linspace(params.momentum[0], params.momentum[1], niter - nBatches))

    Nsum = min(Nchan, 7)  # how many channels to extend out the waveform in mexgetspikes
    # lots of parameters passed into the CUDA scripts
    Params = np.array([
        NT, Nfilt, params.Th[0], nInnerIter, nt0, Nnearest,
        Nrank, params.lam, pmi[0], Nchan, NchanNear, params.nt0min, 1,
        Nsum, NrankPC, params.Th[0]], dtype=np.float64)

    # W0 has to be ordered like this
    W0 = cp.transpose(cp.atleast_3d(cp.asarray(wPCA, dtype=np.float64, order='F')), [0, 2, 1])

    # initialize the list of channels each template lives on
    iList = cp.zeros((Nnearest, Nfilt), dtype=np.int32, order='F')

    # initialize average number of spikes per batch for each template
    nsp = cp.zeros((0, 1), dtype=np.float64, order='F')

    # this flag starts 0, is set to 1 later
    Params[12] = 0

    # kernels for subsample alignment
    Ka, Kb = getKernels(params)

    p1 = .95  # decay of nsp estimate in each batch

    ntot = 0
    # this keeps track of dropped templates for debugging purposes
    ndrop = np.zeros(2, dtype=np.float32, order='F')

    # this is the minimum firing rate that all templates must maintain, or be dropped
    m0 = params.minFR * params.NT / params.fs

    # allocate variables when switching to extraction phase
    # this holds spike times, clusters and other info per spike
    st3 = []  # cp.zeros((int(1e7), 5), dtype=np.float32, order='F')

    # these ones store features per spike
    # Nnearest is the number of nearest templates to store features for
    fW = []  # zeros(Nnearest, 1e7, 'single')
    # NchanNear is the number of nearest channels to take PC features from
    fWpc = []  # zeros(NchanNear, Nrank, 1e7, 'single')

    if 'st3' not in ir:
        for ibatch in tqdm(range(niter), desc="Optimizing templates"):
            # korder is the index of the batch at this point in the schedule
            korder = int(irounds[ibatch])
            # k is the index of the batch in absolute terms
            k = int(isortbatches[korder])

            if ibatch > niter - nBatches - 1 and korder == nhalf:
                # this is required to revert back to the template states in the middle of the
                # batches
                W, dWU = ir.W, ir.dWU
                logger.debug('Reverted back to middle timepoint.')

            if ibatch < niter - nBatches:
                # obtained pm for this batch
                Params[8] = float(pmi[ibatch])
                pm = pmi[ibatch] * ones((Nfilt,), dtype=np.float64, order='F')

            # loading a single batch (same as everywhere)
            offset = Nchan * batchstart[k]
            dat = proc.flat[offset:offset + NT * Nchan].reshape((-1, Nchan), order='F')
            dataRAW = cp.asarray(dat, dtype=np.float32) / params.scaleproc

            if ibatch == 0:
                # only on the first batch, we first get a new set of spikes from the residuals,
                # which in this case is the unmodified data because we start with no templates
                # CUDA function to get spatiotemporal clips from spike detections
                dWU, cmap = mexGetSpikes2(Params, dataRAW, wTEMP, iC)

                dWU = cp.asarray(dWU, dtype=np.float64, order='F')

                # project these into the wPCA waveforms
                dWU = cp.reshape(
                    cp.dot(wPCAd, cp.dot(wPCAd.T, dWU.reshape((dWU.shape[0], -1), order='F'))),
                    dWU.shape, order='F')

                # initialize the low-rank decomposition with standard waves
                W = W0[:, cp.ones(dWU.shape[2], dtype=np.int32), :]
                Nfilt = W.shape[1]  # update the number of filters/templates
                if nsp.size < Nfilt:
                    nsp = cp.zeros(Nfilt, dtype=np.float64, order='F')
                # initialize the number of spikes for new templates with the minimum allowed value,
                # so it doesn't get thrown back out right away
                nsp[:Nfilt] = m0
                Params[1] = Nfilt  # update in the CUDA parameters

            if flag_resort:
                # this is a flag to resort the order of the templates according to best peak
                # channel
                # this is important in order to have cohesive memory requests from the GPU RAM
                # max channel (either positive or negative peak)
                iW = cp.argmax(cp.abs(dWU[nt0min - 1, :, :]), axis=0)
                # iW = int32(squeeze(iW))

                isort = cp.argsort(iW)  # sort by max abs channel
                iW = iW[isort]
                W = W[:, isort, :]  # user ordering to resort all the other template variables
                dWU = dWU[:, :, isort]
                nsp = nsp[isort]

            # decompose dWU by svd of time and space (via covariance matrix of 61 by 61 samples)
            # this uses a "warm start" by remembering the W from the previous iteration
            W, U, mu = mexSVDsmall2(Params, dWU, W, iC, iW, Ka, Kb)

            # UtU is the gram matrix of the spatial components of the low-rank SVDs
            # it tells us which pairs of templates are likely to "interfere" with each other
            # such as when we subtract off a template
            # this needs to change (but I don't know why!)
            UtU, maskU, _ = getMeUtU(iW, iC, mask, Nnearest, Nchan)

            # main CUDA function in the whole codebase. does the iterative template matching
            # based on the current templates, gets features for these templates if requested
            # (featW, featPC),
            # gets scores for the template fits to each spike (vexp), outputs the average of
            # waveforms assigned to each cluster (dWU0),
            # and probably a few more things I forget about
            st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp = mexMPnu8(
                Params, dataRAW, U, W, mu, iC, iW, UtU, iList, wPCA)

            # Sometimes nsp can get transposed (think this has to do with it being
            # a single element in one iteration, to which elements are added
            # nsp, nsp0, and pm must all be row vectors (Nfilt x 1), so force nsp
            # to be a row vector.
            # nsp = cp.atleast_2d(nsp)
            # nsprow, nspcol = nsp.shape
            # if nsprow < nspcol:
            #     nsp = nsp.T
            nsp = nsp.squeeze()

            # updates the templates as a running average weighted by recency
            # since some clusters have different number of spikes, we need to apply the
            # exp(pm) factor several times, and fexp is the resulting update factor
            # for each template
            fexp = np.exp(nsp0 * cp.log(pm[:Nfilt]))
            fexp = cp.reshape(fexp, (1, 1, -1), order='F')
            dWU = dWU * fexp + (1 - fexp) * (dWU0 / cp.reshape(
                cp.maximum(1, nsp0), (1, 1, -1), order='F'))

            # nsp just gets updated according to the fixed factor p1
            nsp = nsp * p1 + (1 - p1) * nsp0

            if ibatch == niter - nBatches - 1:
                # if we reached this point, we need to disable secondary template updates
                # like dropping, and adding new templates. We need to memorize the state of the
                # templates at this timepoint, and set the processing mode to "extraction and
                # tracking"

                flag_resort = 0  # no need to resort templates by channel any more
                # flag_final = 1  # this is the "final" pass

                # final clean up, triage templates one last time
                W, U, dWU, mu, nsp, ndrop = triageTemplates2(
                    params, iW, C2C, W, U, dWU, mu, nsp, ndrop)

                # final number of templates
                Nfilt = W.shape[1]
                Params[1] = Nfilt

                # final covariance matrix between all templates
                WtW, iList = getMeWtW(W, U, Nnearest)

                # iW is the final channel assigned to each template
                iW = cp.argmax(cp.abs(dWU[nt0min - 1, :, :]), axis=0)

                # extract ALL features on the last pass
                Params[12] = 2  # this is a flag to output features (PC and template features)

                # different threshold on last pass?
                Params[2] = params.Th[-1]  # usually the threshold is much lower on the last pass

                # memorize the state of the templates
                ir.W, ir.dWU, ir.U, ir.mu = W, dWU, U, mu
                ir.Wraw = cp.zeros(
                    (U.shape[0], W.shape[0], U.shape[1]), dtype=np.float64, order='F')
                for n in range(U.shape[1]):
                    # temporarily use U rather Urot until I have a chance to test it
                    ir.Wraw[:, :, n] = mu[n] * cp.dot(U[:, n, :], W[:, n, :].T)

            if ibatch < niter - nBatches - 1:
                # during the main "learning" phase of fitting a model
                if ibatch % 5 == 0:
                    # this drops templates based on spike rates and/or similarities to
                    # other templates
                    W, U, dWU, mu, nsp, ndrop = triageTemplates2(
                        params, iW, C2C, W, U, dWU, mu, nsp, ndrop)

                Nfilt = W.shape[1]  # update the number of filters
                Params[1] = Nfilt

                # this adds new templates if they are detected in the residual
                dWU0, cmap = mexGetSpikes2(Params, drez, wTEMP, iC)

                if dWU0.shape[2] > 0:
                    # new templates need to be integrated into the same format as all templates
                    # apply PCA for smoothing purposes
                    dWU0 = cp.reshape(cp.dot(wPCAd, cp.dot(
                        wPCAd.T, dWU0.reshape(
                            (dWU0.shape[0], dWU0.shape[1] * dWU0.shape[2]), order='F'))),
                        dWU0.shape, order='F')
                    dWU = cp.concatenate((dWU, dWU0), axis=2)

                    m = dWU0.shape[2]

                    if W.shape[1] < Nfilt + m:
                        W = cp.concatenate(
                            (W, cp.zeros(
                                (W.shape[0], Nfilt + m - W.shape[1], W.shape[2]),
                                dtype=W.dtype)), axis=1)

                    # initialize temporal components of waveforms
                    W[:, Nfilt:Nfilt + m, :] = W0[:, cp.ones(m, dtype=np.int32), :]

                    # initialize the number of spikes with the minimum allowed
                    nsp[Nfilt:Nfilt + m] = params.minFR * NT / params.fs
                    # initialize the amplitude of this spike with a lowish number
                    mu[Nfilt:Nfilt + m] = 10

                    # if the number of filters exceed the maximum allowed, clip it
                    Nfilt = min(params.Nfilt, W.shape[1])
                    Params[1] = Nfilt

                    W = W[:, :Nfilt, :]  # remove any new filters over the maximum allowed
                    dWU = dWU[:, :, :Nfilt]  # remove any new filters over the maximum allowed
                    nsp = nsp[:Nfilt]  # remove any new filters over the maximum allowed
                    mu = mu[:Nfilt]  # remove any new filters over the maximum allowed

            if ibatch > niter - nBatches - 1:
                # during the final extraction pass, this keesp track of all spikes and features

                # we memorize the spatio-temporal decomposition of the waveforms at this batch
                # this is currently only used in the GUI to provide an accurate reconstruction
                # of the raw data at this time
                ir.WA[..., k] = cp.asnumpy(W)
                ir.UA[..., k] = cp.asnumpy(U)
                ir.muA[..., k] = cp.asnumpy(mu)

                # we carefully assign the correct absolute times to spikes found in this batch
                ioffset = params.ntbuff - 1
                if k == 0:
                    ioffset = 0  # the first batch is special (no pre-buffer)

                toff = nt0min + t0 - ioffset + (NT - params.ntbuff) * (k - 1)
                st = toff + st0

                # irange = np.arange(ntot, ntot+x0.size)  # spikes and features go into
                # these indices

            #         if ntot + x0.size > st3.shape[0]:
            #            # if we exceed the original allocated memory, double the allocated sizes
            #            fW[:, 2 * st3.shape[0]] = 0
            #            fWpc[:, :, 2 * st3.shape[0]] = 0
            #            st3[2 * st3.shape[0] - 1, 0] = 0

                st3.append(np.c_[
                    cp.asnumpy(st),  # spike times
                    cp.asnumpy(id0),  # spike clusters (0-indexing)
                    cp.asnumpy(x0),  # template amplitudes
                    cp.asnumpy(vexp),  # residual variance of this spike
                    korder * np.ones(st.size),  # batch from which this spike was found
                ])
            #         st3[irange, 1] = id0  # spike clusters (1-indexing)
            #         st3[irange, 2] = x0  # template amplitudes
            #         st3[irange, 3] = vexp  # residual variance of this spike
            #         st3[irange, 4] = korder  # batch from which this spike was found

            #         fW[:, irange] = featW  # template features for this batch
            #         fWpc[:, :, irange] = featPC  # PC features
                fW.append(cp.asnumpy(featW))
                fWpc.append(cp.asnumpy(featPC))

                ntot = ntot + x0.size  # keeps track of total number of spikes so far

            if ibatch == niter - nBatches - 1:
                # # allocate variables when switching to extraction phase
                # # this holds spike times, clusters and other info per spike
                # st3 = []  # cp.zeros((int(1e7), 5), dtype=np.float32, order='F')

                # these next three store the low-d template decompositions
                ir.WA = np.zeros((nt0, Nfilt, Nrank, nBatches), dtype=np.float32, order='F')
                ir.UA = np.zeros((Nchan, Nfilt, Nrank, nBatches), dtype=np.float32, order='F')
                ir.muA = np.zeros((Nfilt, nBatches), dtype=np.float32, order='F')

                # # these next three store the low-d template decompositions
                # ir.WA = []  # zeros(nt0, Nfilt, Nrank,nBatches,  'single')
                # ir.UA = []  # zeros(Nchan, Nfilt, Nrank,nBatches,  'single')
                # ir.muA = []  # zeros(Nfilt, nBatches,  'single')

                # # these ones store features per spike
                # # Nnearest is the number of nearest templates to store features for
                # fW = []  # zeros(Nnearest, 1e7, 'single')
                # # NchanNear is the number of nearest channels to take PC features from
                # fWpc = []  # zeros(NchanNear, Nrank, 1e7, 'single')

            if ibatch % 100 == 0:
                # this is some of the relevant diagnostic information to be printed during training
                logger.info(
                    ('%d / %d batches, %d units, nspks: %2.4f, mu: %2.4f, '
                     'nst0: %d, merges: %2.4f, %2.4f'),
                    ibatch, niter, Nfilt, nsp.sum(), median(mu), st0.size, ndrop)

            # discards the unused portion of the arrays
            #st3 = st3(1:ntot, :)
            #fW = fW(:, 1:ntot)
            #fWpc = fWpc(:,:, 1:ntot)

            free_gpu_memory()

        # just display the total number of spikes
        logger.info("Found %d spikes.", ntot)

        # Save results to the ctx.intermediate object.
        ir.st3 = np.concatenate(st3, axis=0)

        # the similarity score between templates is simply the correlation,
        # taken as the max over several consecutive time delays
        ir.simScore = cp.asnumpy(cp.max(WtW, axis=2))

        fWa = np.concatenate(fW, axis=-1)
        fWpca = np.concatenate(fWpc, axis=-1)

        # the template features are stored in cProj, like in Kilosort1
        ir.cProj = fWa.T
        # the neihboring templates idnices are stored in iNeigh
        ir.iNeigh = cp.asnumpy(iList)

        #  permute the PC projections in the right order
        ir.cProjPC = np.transpose(fWpca, (2, 1, 0))
        # iNeighPC keeps the indices of the channels corresponding to the PC features
        ir.iNeighPC = cp.asnumpy(iC[:, iW])

        # These are the variables set by the code above:
        ctx.save(
            st3=ir.st3,
            simScore=ir.simScore,
            cProj=ir.cProj,
            cProjPC=ir.cProjPC,
            iNeigh=ir.iNeigh,
            iNeighPC=ir.iNeighPC,
            WA=ir.WA,
            UA=ir.UA,
            W=ir.W,
            U=ir.U,
            dWU=ir.dWU,
            mu=ir.mu,
        )

    if 'W_a' not in ir:
        # this whole next block is just done to compress the compressed templates
        # we separately svd the time components of each template, and the spatial components
        # this also requires a careful decompression function, available somewhere in the GUI code
        nKeep = min(Nchan * 3, 20)  # how many PCs to keep
        W_a = np.zeros((nt0 * Nrank, nKeep, Nfilt), dtype=np.float32)
        W_b = np.zeros((nBatches, nKeep, Nfilt), dtype=np.float32)
        U_a = np.zeros((Nchan * Nrank, nKeep, Nfilt), dtype=np.float32)
        U_b = np.zeros((nBatches, nKeep, Nfilt), dtype=np.float32)

        for j in tqdm(range(Nfilt), desc='Compressing templates'):
            # do this for every template separately
            WA = np.reshape(ir.WA[:, j, ...], (-1, nBatches), order='F')
            # svd on the GPU was faster for this, but the Python randomized CPU version
            # might be faster still
            # WA = gpuArray(WA)
            A, B, C = svdecon_cpu(WA)
            # W_a times W_b results in a reconstruction of the time components
            W_a[:, :, j] = np.dot(A[:, :nKeep], B[:nKeep, :nKeep])
            W_b[:, :, j] = C[:, :nKeep]

            UA = np.reshape(ir.UA[:, j, ...], (-1, nBatches), order='F')
            # UA = gpuArray(UA)
            A, B, C = svdecon_cpu(UA)
            # U_a times U_b results in a reconstruction of the time components
            U_a[:, :, j] = np.dot(A[:, :nKeep], B[:nKeep, :nKeep])
            U_b[:, :, j] = C[:, :nKeep]

        ctx.save(
            W_a=W_a,
            W_b=W_b,
            U_a=U_a,
            U_b=U_b,
        )

    logger.info('Finished compressing time-varying templates.')
