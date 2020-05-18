import logging
from math import ceil

import numpy as np
import cupy as cp
from tqdm import tqdm

from .preprocess import my_min, my_sum
from .cptools import svdecon, zscore, ones
from .utils import Bunch, get_cuda

logger = logging.getLogger(__name__)


def getClosestChannels(probe, sigma, NchanClosest):
    # this function outputs the closest channels to each channel,
    # as well as a Gaussian-decaying mask as a function of pairwise distances
    # sigma is the standard deviation of this Gaussian-mask

    # compute distances between all pairs of channels
    xc = cp.asarray(probe.xc, dtype=np.float32, order='F')
    yc = cp.asarray(probe.yc, dtype=np.float32, order='F')
    C2C = (xc[:, np.newaxis] - xc) ** 2 + (yc[:, np.newaxis] - yc) ** 2
    C2C = cp.sqrt(C2C)
    Nchan = C2C.shape[0]

    # sort distances
    isort = cp.argsort(C2C, axis=0)

    # take NchanCLosest neighbors for each primary channel
    iC = isort[:NchanClosest, :]

    # in some cases we want a mask that decays as a function of distance between pairs of channels
    # this is an awkward indexing to get the corresponding distances
    ix = iC + cp.arange(0, Nchan ** 2, Nchan)
    mask = cp.exp(-C2C.T.ravel()[ix] ** 2 / (2 * sigma ** 2))

    # masks should be unit norm for each channel
    mask = mask / cp.sqrt(1e-3 + cp.sum(mask ** 2, axis=0))

    return iC, mask, C2C


def isolated_peaks_new(S1, params):
    """
    takes a matrix of timepoints by channels S1
    outputs threshold crossings that are relatively isolated from other peaks
    outputs row, column and magnitude of the threshold crossing
    """
    S1 = cp.asarray(S1)

    # finding the local minimum in a sliding window within plus/minus loc_range extent
    # across time and across channels
    smin = my_min(S1, params.loc_range, [0, 1])

    # the peaks are samples that achieve this local minimum, AND have negativities less
    # than a preset threshold
    peaks = (S1 < smin + 1e-3) & (S1 < params.spkTh)

    # only take local peaks that are isolated from other local peaks
    # if there is another local peak close by, this sum will be at least 2
    sum_peaks = my_sum(peaks, params.long_range, [0, 1])
    # set to 0 peaks that are not isolated, and multiply with the voltage values
    peaks = peaks * (sum_peaks < 1.2) * S1

    # exclude temporal buffers
    peaks[:params.nt0, :] = 0
    peaks[-params.nt0:, :] = 0

    # find the non-zero peaks, and take their amplitudes
    col, row = cp.nonzero(peaks.T)
    # invert the sign of the amplitudes
    mu = -peaks[row, col]

    return row, col, mu


def get_SpikeSample(dataRAW, row, col, params):
    """
    given a batch of data (time by channels), and some time (row) and channel (col) indices for
    spikes, this function returns the 1D time clips of voltage around those spike times
    """
    nT, nChan = dataRAW.shape

    # times around the peak to consider
    dt = cp.arange(params.nt0)

    # the negativity is expected at nt0min, so we align the detected peaks there
    dt = -params.nt0min + dt

    # temporal indices (awkward way to index into full matrix of data)
    indsT = row + dt[:, np.newaxis] + 1  # broadcasting
    indsC = col

    indsC[indsC < 0] = 0  # anything that's out of bounds just gets set to the limit
    indsC[indsC >= nChan] = nChan - 1  # only needed for channels not time (due to time buffer)

    indsT = cp.transpose(cp.atleast_3d(indsT), [0, 2, 1])
    indsC = cp.transpose(cp.atleast_3d(indsC), [2, 0, 1])

    # believe it or not, these indices grab just the right timesamples forour spikes
    ix = indsT + indsC * nT

    # grab the data and reshape it appropriately (time samples  by channels by num spikes)
    clips = dataRAW.T.ravel()[ix[:, 0, :]].reshape((dt.size, row.size), order='F')  # HERE
    return clips


def extractPCfromSnippets(proc, probe=None, params=None, Nbatch=None):
    # extracts principal components for 1D snippets of spikes from all channels
    # loads a subset of batches to find these snippets

    NT = params.NT
    nPCs = params.nPCs
    Nchan = probe.Nchan

    batchstart = np.arange(0, NT * Nbatch + 1, NT).astype(np.int64)

    # extract the PCA projections
    # initialize the covariance of single-channel spike waveforms
    CC = cp.zeros(params.nt0, dtype=np.float32)

    # from every 100th batch
    for ibatch in range(0, Nbatch, 100):
        offset = Nchan * batchstart[ibatch]
        dat = proc.flat[offset:offset + NT * Nchan].reshape((-1, Nchan), order='F')
        if dat.shape[0] == 0:
            continue

        # move data to GPU and scale it back to unit variance
        dataRAW = cp.asarray(dat, dtype=np.float32) / params.scaleproc

        # find isolated spikes from each batch
        row, col, mu = isolated_peaks_new(dataRAW, params)

        # for each peak, get the voltage snippet from that channel
        c = get_SpikeSample(dataRAW, row, col, params)

        # scale covariance down by 1,000 to maintain a good dynamic range
        CC = CC + cp.dot(c, c.T) / 1e3

    # the singular vectors of the covariance matrix are the PCs of the waveforms
    U, Sv, V = svdecon(CC)

    wPCA = U[:, :nPCs]  # take as many as needed

    # adjust the arbitrary sign of the first PC so its negativity is downward
    wPCA[:, 0] = -wPCA[:, 0] * cp.sign(wPCA[20, 0])

    return wPCA


def sortBatches2(ccb0):
    # takes as input a matrix of nBatches by nBatches containing
    # dissimilarities.
    # outputs a matrix of sorted batches, and the sorting order, such that
    # ccb1 = ccb0(isort, isort)

    # put this matrix on the GPU
    ccb0 = cp.asarray(ccb0, order='F')

    # compute its svd on the GPU (this might also be fast enough on CPU)
    u, s, v = svdecon(ccb0)
    # HACK: consistency with MATLAB
    u = u * cp.sign(u[0, 0])
    v = v * cp.sign(u[0, 0])

    # initialize the positions xs of the batch embeddings to be very small but proportional to
    # the first PC
    xs = .01 * u[:, 0] / cp.std(u[:, 0], ddof=1)

    # 200 iterations of gradient descent should be enough
    niB = 200

    # this learning rate should usually work fine, since it scales with the average gradient
    # and ccb0 is z-scored
    eta = 1
    for k in tqdm(range(niB), desc="Sorting %d batches" % ccb0.shape[0]):
        # euclidian distances between 1D embedding positions
        ds = (xs - xs[:, np.newaxis]) ** 2
        # the transformed distances go through this function
        W = cp.log(1 + ds)

        # the error is the difference between ccb0 and W
        err = ccb0 - W

        # ignore the mean value of ccb0
        err = err - cp.mean(err, axis=0)

        # backpropagate the gradients
        err = err / (1 + ds)
        err2 = err * (xs[:, np.newaxis] - xs)
        D = cp.mean(err2, axis=1)  # one half of the gradients is along this direction
        E = cp.mean(err2, axis=0)  # the other half is along this direction
        # we don't need to worry about the gradients for the diagonal because those are 0

        # final gradients for the embedding variable
        dx = -D + E.T

        # take a gradient step
        xs = xs - eta * dx

    # sort the embedding positions xs
    isort = cp.argsort(xs, axis=0)

    # sort the matrix of dissimilarities
    ccb1 = ccb0[isort, :][:, isort]

    return ccb1, isort


def initializeWdata2(call, uprojDAT, Nchan, nPCs, Nfilt, iC):
    # this function initializes cluster means for the fast kmeans per batch
    # call are time indices for the spikes
    # uprojDAT are features projections (Nfeatures by Nspikes)
    # some more parameters need to be passed in from the main workspace

    # pick random spikes from the sample
    # WARNING: replace ceil by warning because this is a random index, and 0/1 indexing
    # discrepancy between Python and MATLAB.
    irand = np.floor(np.random.rand(Nfilt) * uprojDAT.shape[1]).astype(np.int32)

    W = cp.zeros((nPCs, Nchan, Nfilt), dtype=np.float32)

    for t in range(Nfilt):
        ich = iC[:, call[irand[t]]]  # the channels on which this spike lives
        # for each selected spike, get its features
        W[:, ich, t] = uprojDAT[:, irand[t]].reshape(W[:, ich, t].shape, order='F')

    W = W.reshape((-1, Nfilt), order='F')  # HERE
    # add small amount of noise in case we accidentally picked the same spike twice
    W = W + .001 * cp.random.normal(size=W.shape).astype(np.float32)
    mu = cp.sqrt(cp.sum(W ** 2, axis=0))  # get the mean of the template
    W = W / (1e-5 + mu)  # and normalize the template
    W = W.reshape((nPCs, Nchan, Nfilt), order='F')  # HERE
    nW = (W[0, ...] ** 2)  # squared amplitude of the first PC feture
    W = W.reshape((nPCs * Nchan, Nfilt), order='F')  # HERE
    # determine biggest channel according to the amplitude of the first PC
    Wheights = cp.argmax(nW, axis=0)

    return W, mu, Wheights, irand


def mexThSpkPC(Params, dataRAW, wPCA, iC):
    code, constants = get_cuda('mexThSpkPC')
    Nthreads = constants.Nthreads
    maxFR = constants.maxFR

    NT, Nchan, NchanNear, nt0, nt0min, spkTh, NrankPC = Params
    NT = int(NT)
    Nchan = int(Nchan)

    # Input GPU arrays.
    d_Params = cp.asarray(Params, dtype=np.float64, order='F')
    d_data = cp.asarray(dataRAW, dtype=np.float32, order='F')
    d_W = cp.asarray(wPCA, dtype=np.float32, order='F')
    d_iC = cp.asarray(iC, dtype=np.int32, order='F')

    # New GPU arrays.
    d_dout = cp.zeros((Nchan, NT), dtype=np.float32, order='F')
    d_dmax = cp.zeros((Nchan, NT), dtype=np.float32, order='F')
    d_st = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_id = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_counter = cp.zeros(1, dtype=np.int32, order='F')

    # filter the data with the temporal templates
    Conv1D = cp.RawKernel(code, 'Conv1D')
    Conv1D((Nchan,), (Nthreads,), (d_Params, d_data, d_W, d_dout))

    # get the max of the data
    max1D = cp.RawKernel(code, 'max1D')
    max1D((Nchan,), (Nthreads,), (d_Params, d_dout, d_dmax))

    # take max across nearby channels
    maxChannels = cp.RawKernel(code, 'maxChannels')
    maxChannels(
        (int(NT // Nthreads),), (Nthreads,),
        (d_Params, d_dout, d_dmax, d_iC, d_st, d_id, d_counter))

    # move d_x to the CPU
    minSize = 1
    minSize = min(maxFR, int(d_counter[0]))

    d_featPC = cp.zeros((NrankPC * NchanNear, minSize), dtype=np.float32, order='F')

    d_id2 = cp.zeros(minSize, dtype=np.int32, order='F')

    if (minSize > 0):
        computeProjections = cp.RawKernel(code, 'computeProjections')
        computeProjections(
            (minSize,), (NchanNear, NrankPC), (d_Params, d_data, d_iC, d_st, d_id, d_W, d_featPC))

    # TODO: check that the copy occurs on the GPU only
    d_id2[:] = d_id[:minSize]

    # Free memory.
    del d_st, d_id, d_counter, d_Params, d_dmax, d_dout
    # free_gpu_memory()

    return d_featPC, d_id2


def extractPCbatch2(proc, params, probe, wPCA, ibatch, iC, Nbatch):
    # this function finds threshold crossings in the data using
    # projections onto the pre-determined principal components
    # wPCA is number of time samples by number of PCs
    # ibatch is a scalar indicating which batch to analyze
    # iC is NchanNear by Nchan, indicating for each channel the nearest
    # channels to it

    nt0min = params.nt0min
    spkTh = params.ThPre
    nt0, NrankPC = wPCA.shape
    NT, Nchan = params.NT, probe.Nchan

    # starts with predefined PCA waveforms
    wPCA = wPCA[:, :3]

    NchanNear = iC.shape[0]

    # batches start at these timepoints
    batchstart = np.arange(0, NT * Nbatch + 1, NT).astype(np.int64)

    offset = Nchan * batchstart[ibatch]
    dat = proc.flat[offset:offset + NT * Nchan].reshape((-1, Nchan), order='F')
    dataRAW = cp.asarray(dat, dtype=np.float32) / params.scaleproc

    # another Params variable to take all our parameters into the C++ code
    Params = [NT, Nchan, NchanNear, nt0, nt0min, spkTh, NrankPC]

    # call a CUDA function to do the hard work
    # returns a matrix of features uS, as well as the center channels for each spike
    uS, idchan = mexThSpkPC(Params, dataRAW, wPCA, iC)

    return uS, idchan


def mexClustering2(Params, uproj, W, mu, call, iMatch, iC):

    code, _ = get_cuda('mexClustering2')

    Nspikes = int(Params[0])
    NrankPC = int(Params[1])
    Nfilters = int(Params[2])
    NchanNear = int(Params[6])
    Nchan = int(Params[7])

    d_Params = cp.asarray(Params, dtype=np.float64, order='F')
    d_uproj = cp.asarray(uproj, dtype=np.float32, order='F')
    d_W = cp.asarray(W, dtype=np.float32, order='F')
    d_mu = cp.asarray(mu, dtype=np.float32, order='F')
    d_call = cp.asarray(call, dtype=np.int32, order='F')
    d_iC = cp.asarray(iC, dtype=np.int32, order='F')
    d_iMatch = cp.asarray(iMatch, dtype=np.bool, order='F')

    d_dWU = cp.zeros((NrankPC * Nchan, Nfilters), dtype=np.float32, order='F')
    d_cmax = cp.zeros((Nspikes, Nfilters), dtype=np.float32, order='F')
    d_id = cp.zeros(Nspikes, dtype=np.int32, order='F')
    d_x = cp.zeros(Nspikes, dtype=np.float32, order='F')
    d_nsp = cp.zeros(Nfilters, dtype=np.int32, order='F')
    d_V = cp.zeros(Nfilters, dtype=np.float32, order='F')

    # get list of cmaxes for each combination of neuron and filter
    computeCost = cp.RawKernel(code, 'computeCost')
    computeCost(
        (Nfilters,), (1024,), (d_Params, d_uproj, d_mu, d_W, d_iMatch, d_iC, d_call, d_cmax))

    # loop through cmax to find best template
    bestFilter = cp.RawKernel(code, 'bestFilter')
    bestFilter((40,), (256,), (d_Params, d_iMatch, d_iC, d_call, d_cmax, d_id, d_x))

    # average all spikes for same template -- ORIGINAL
    average_snips = cp.RawKernel(code, 'average_snips')
    average_snips(
        (Nfilters,), (NrankPC, NchanNear), (d_Params, d_iC, d_call, d_id, d_uproj, d_cmax, d_dWU))

    count_spikes = cp.RawKernel(code, 'count_spikes')
    count_spikes((7,), (256,), (d_Params, d_id, d_nsp, d_x, d_V))

    del d_Params, d_V

    return d_dWU, d_id, d_x, d_nsp, d_cmax


def mexDistances2(Params, Ws, W, iMatch, iC, Wh, mus, mu):
    code, _ = get_cuda('mexDistances2')

    Nspikes = int(Params[0])
    Nfilters = int(Params[2])

    d_Params = cp.asarray(Params, dtype=np.float64, order='F')

    d_Ws = cp.asarray(Ws, dtype=np.float32, order='F')
    d_W = cp.asarray(W, dtype=np.float32, order='F')
    d_iMatch = cp.asarray(iMatch, dtype=np.bool, order='F')
    d_iC = cp.asarray(iC, dtype=np.int32, order='F')
    d_Wh = cp.asarray(Wh, dtype=np.int32, order='F')
    d_mu = cp.asarray(mu, dtype=np.float32, order='F')
    d_mus = cp.asarray(mus, dtype=np.float32, order='F')

    d_cmax = cp.zeros(Nspikes * Nfilters, dtype=np.float32, order='F')
    d_id = cp.zeros(Nspikes, dtype=np.int32, order='F')
    d_x = cp.zeros(Nspikes, dtype=np.float32, order='F')

    # get list of cmaxes for each combination of neuron and filter
    computeCost = cp.RawKernel(code, 'computeCost')
    computeCost(
        (Nfilters,), (1024,), (d_Params, d_Ws, d_mus, d_W, d_mu, d_iMatch, d_iC, d_Wh, d_cmax))

    # loop through cmax to find best template
    bestFilter = cp.RawKernel(code, 'bestFilter')
    bestFilter((40,), (256,), (d_Params, d_iMatch, d_Wh, d_cmax, d_mus, d_id, d_x))

    del d_Params, d_cmax

    return d_id, d_x


def clusterSingleBatches(ctx):
    """
    outputs an ordering of the batches according to drift
    for each batch, it extracts spikes as threshold crossings and clusters them with kmeans
    the resulting cluster means are then compared for all pairs of batches, and a dissimilarity
    score is assigned to each pair
    the matrix of similarity scores is then re-ordered so that low dissimilaity is along
    the diagonal
    """
    Nbatch = ctx.intermediate.Nbatch
    params = ctx.params
    probe = ctx.probe
    raw_data = ctx.raw_data
    ir = ctx.intermediate
    proc = ir.proc

    if not params.reorder:
        # if reordering is turned off, return consecutive order
        iorig = np.arange(Nbatch)
        return iorig, None, None

    nPCs = params.nPCs
    Nfilt = ceil(probe.Nchan / 2)

    # extract PCA waveforms pooled over channels
    wPCA = extractPCfromSnippets(proc, probe=probe, params=params, Nbatch=Nbatch)

    Nchan = probe.Nchan
    niter = 10  # iterations for k-means. we won't run it to convergence to save time

    nBatches = Nbatch
    NchanNear = min(Nchan, 2 * 8 + 1)

    # initialize big arrays on the GPU to hold the results from each batch
    # this holds the unit norm templates
    Ws = cp.zeros((nPCs, NchanNear, Nfilt, nBatches), dtype=np.float32, order='F')
    # this holds the scalings
    mus = cp.zeros((Nfilt, nBatches), dtype=np.float32, order='F')
    # this holds the number of spikes for that cluster
    ns = cp.zeros((Nfilt, nBatches), dtype=np.float32, order='F')
    # this holds the center channel for each template
    Whs = ones((Nfilt, nBatches), dtype=np.int32, order='F')

    i0 = 0
    NrankPC = 3  # I am not sure if this gets used, but it goes into the function

    # return an array of closest channels for each channel
    iC = getClosestChannels(probe, params.sigmaMask, NchanNear)[0]

    for ibatch in tqdm(range(nBatches), desc="Clustering spikes"):

        # extract spikes using PCA waveforms
        uproj, call = extractPCbatch2(
            proc, params, probe, wPCA, min(nBatches - 2, ibatch), iC, Nbatch)

        if cp.sum(cp.isnan(uproj)) > 0:
            break  # I am not sure what case this safeguards against....

        if uproj.shape[1] > Nfilt:

            # this initialize the k-means
            W, mu, Wheights, irand = initializeWdata2(call, uproj, Nchan, nPCs, Nfilt, iC)

            # Params is a whole bunch of parameters sent to the C++ scripts inside a float64 vector
            Params = [uproj.shape[1], NrankPC, Nfilt, 0, W.shape[0], 0, NchanNear, Nchan]

            for i in range(niter):

                Wheights = Wheights.reshape((1, 1, -1), order='F')
                iC = cp.atleast_3d(iC)

                # we only compute distances to clusters on the same channels
                # this tells us which spikes and which clusters might match
                iMatch = cp.min(cp.abs(iC - Wheights), axis=0) < .1

                # get iclust and update W
                # CUDA script to efficiently compute distances for pairs in which iMatch is 1
                dWU, iclust, dx, nsp, dV = mexClustering2(Params, uproj, W, mu, call, iMatch, iC)

                dWU = dWU / (1e-5 + nsp.T)  # divide the cumulative waveform by the number of spike

                mu = cp.sqrt(cp.sum(dWU ** 2, axis=0))  # norm of cluster template
                W = dWU / (1e-5 + mu)  # unit normalize templates

                W = W.reshape((nPCs, Nchan, Nfilt), order='F')
                nW = W[0, ...] ** 2  # compute best channel from the square of the first PC feature
                W = W.reshape((Nchan * nPCs, Nfilt), order='F')

                Wheights = cp.argmax(nW, axis=0)  # the new best channel of each cluster template

            # carefully keep track of cluster templates in dense format
            W = W.reshape((nPCs, Nchan, Nfilt), order='F')
            W0 = cp.zeros((nPCs, NchanNear, Nfilt), dtype=np.float32, order='F')
            for t in range(Nfilt):
                W0[..., t] = W[:, iC[:, Wheights[t]], t].squeeze()
            # I don't really know why this needs another normalization
            W0 = W0 / (1e-5 + cp.sum(cp.sum(W0 ** 2, axis=0)[np.newaxis, ...], axis=1) ** .5)

        # if a batch doesn't have enough spikes, it gets the cluster templates of the previous batc
        if 'W0' in locals():
            Ws[..., ibatch] = W0
            mus[:, ibatch] = mu
            ns[:, ibatch] = nsp
            Whs[:, ibatch] = Wheights.astype(np.int32)
        else:
            logger.warning('Data batch #%d only had %d spikes.', ibatch, uproj.shape[1])

        i0 = i0 + Nfilt

    # anothr one of these Params variables transporting parameters to the C++ code
    Params = [1, NrankPC, Nfilt, 0, W.shape[0], 0, NchanNear, Nchan]
    # the total number of templates is the number of templates per batch times the number of batch
    Params[0] = Ws.shape[2] * Ws.shape[3]

    # initialize dissimilarity matrix
    ccb = cp.zeros((nBatches, nBatches), dtype=np.float32, order='F')

    for ibatch in tqdm(range(nBatches), desc="Computing distances"):
        # for every batch, compute in parallel its dissimilarity to ALL other batches
        Wh0 = Whs[:, ibatch]  # this one is the primary batch
        W0 = Ws[..., ibatch]
        mu = mus[..., ibatch]

        # embed the templates from the primary batch back into a full, sparse representation
        W = cp.zeros((nPCs, Nchan, Nfilt), dtype=np.float32, order='F')
        for t in range(Nfilt):
            W[:, iC[:, Wh0[t]], t] = cp.atleast_3d(Ws[:, :, t, ibatch])

        # pairs of templates that live on the same channels are potential "matches"
        iMatch = cp.min(cp.abs(iC - Wh0.reshape((1, 1, -1), order='F')), axis=0) < .1

        # compute dissimilarities for iMatch = 1
        iclust, ds = mexDistances2(Params, Ws, W, iMatch, iC, Whs, mus, mu)

        # ds are squared Euclidian distances
        ds = ds.reshape((Nfilt, -1), order='F')  # this should just be an Nfilt-long vector
        ds = cp.maximum(0, ds)

        # weigh the distances according to number of spikes in cluster
        ccb[ibatch, :] = cp.mean(cp.sqrt(ds) * ns, axis=0) / cp.mean(ns, axis=0)

    # ccb = cp.asnumpy(ccb)
    # some normalization steps are needed: zscoring, and symmetrizing ccb
    ccb0 = zscore(ccb, axis=0)
    ccb0 = ccb0 + ccb0.T

    # sort by manifold embedding algorithm
    # iorig is the sorting of the batches
    # ccbsort is the resorted matrix (useful for diagnosing drift)
    ccbsort, iorig = sortBatches2(ccb0)

    logger.info("Finished clustering.")

    return Bunch(iorig=iorig, ccb0=ccb0, ccbsort=ccbsort)
