import cupy as cp
from math import erf, log, sqrt

from .cptools import svdecon
from .cluster import getClosestChannels
from .learn import getKernels, getMeWtW, mexSVDsmall2
from .utils import Bunch


def ccg(st1, st2, nbins, tbin):
    # this function efficiently computes the crosscorrelogram between two sets
    # of spikes (st1, st2), with tbin length each, timelags =  plus/minus nbins
    # and then estimates how refractory the cross-correlogram is, which can be used
    # during merge decisions.

    st1 = cp.sort(st1)  # makes sure spike trains are sorted in increasing order
    st2 = cp.sort(st2)

    dt = nbins * tbin

    T = cp.max(cp.concatenate((st1, st2))) - cp.min(cp.concatenate((st1, st2)))

    # we traverse both spike trains together, keeping track of the spikes in the first
    # spike train that are within dt of spikes in the second spike train

    ilow = 0  # lower bound index
    ihigh = 0  # higher bound index
    j = 0  # index of the considered spike

    K = cp.zeros(2 * nbins + 1)

    # (DEV_NOTES) the while loop below is far too slow as is

    while j <= len(st2) - 1:  # traverse all spikes in the second spike train

        while (ihigh <= len(st1) - 1) and (st1[ihigh] < st2[j] + dt):
            ihigh += 1  # keep increasing higher bound until it's OUTSIDE of dt range

        while (ilow <= len(st1) - 1) and (st1[ilow] <= st2[j] - dt):
            ilow += 1  # keep increasing lower bound until it's INSIDE of dt range

        if ilow > len(st1) - 1:
            break  # break if we exhausted the spikes from the first spike train

        if st1[ilow] > st2[j] + dt:
            # if the lower bound is actually outside of dt range, means we overshot (there were no spikes in range)
            # simply move on to next spike from second spike train
            j += 1
            continue

        for k in range(ilow, ihigh):
            # for all spikes within plus/minus dt range
            ibin = cp.rint((st2[j] - st1[k]) / tbin).astype(int)  # convert ISI to integer

            K[ibin + nbins] += 1

        j += 1

    irange1 = cp.concatenate((cp.arange(1, nbins // 2), cp.arange(3 * nbins // 2, 2 * nbins)))
    irange2 = cp.arange(nbins - 50, nbins - 10)
    irange3 = cp.arange(nbins + 11, nbins + 50)

    # normalize the shoulders by what's expected from the mean firing rates
    # a non-refractive poisson process should yield 1

    Q00 = cp.sum(K[irange1]) / (len(irange1) * tbin * len(st1) * len(st2) / T)
    Q01 = cp.sum(K[irange2]) / (len(irange2) * tbin * len(st1) * len(st2) / T)  # do the same for irange 2
    Q01 = max(Q01, cp.sum(K[irange3]) / (len(irange3) * tbin * len(st1) * len(st2) / T))  # compare to the other shoulder

    R00 = max(cp.mean(K[irange2]), cp.mean(K[irange3]))  # take the biggest shoulder
    R00 = max(R00, cp.mean(K[irange1]))  # compare this to the asymptotic shoulder

    # test the probability that a central area in the autocorrelogram might be refractory
    # test increasingly larger areas of the central CCG

    a = K[nbins]
    K[nbins] = 0

    Qi = cp.zeros(10)
    Ri = cp.zeros(10)

    for i in range(1, 11):
        irange = cp.arange(nbins - i, nbins + i + 1)  # for this central range of the CCG
        Qi0 = cp.sum(K[irange]) / (2 * i * tbin * len(st1) * len(st2) / T)  # compute the normalised ratio as above. this should be 1 if there is no refractoriness
        Qi[i - 1] = Qi0  # save the normalised probability

        n = cp.sum(K[irange]) / 2
        lam = R00 * i

        # log(p) = log(lam) * n - lam - gammaln(n+1)

        # this is tricky: we approximate the Poisson likelihood with a gaussian of equal mean and variance
        # that allows us to integrate the probability that we would see <N spikes in the center of the
        # cross-correlogram from a distribution with mean R00*i spikes

        p = 1 / 2 * (1 + erf((n - lam) / cp.sqrt(2 * lam)))

        Ri[i - 1] = p  # keep track of p for each bin size i

    K[nbins] = a  # restore the center value of the cross-correlogram
    Qin = Qi / Q00  # normalize the normalized refractory index in two different ways
    Qin1 = Qi / Q01

    # (DEV_NOTES) Variables Qin and Qin1 seem redundant?

    return K, Qi, Q00, Q01, Ri


def find_merges(rez, flag):
    # this function merges clusters based on template correlation
    # however, a merge is veto-ed if refractory period violations are introduced

    ops = rez.ops
    dt = 1 / 1000  # step size for CCG binning
    nbins = 500  # number of bins used for cross-correlograms

    # (DEV_NOTES) nbins is not a variable in Marius' code, I include it here to avoid inexplianable, hard-coded constants later

    Xsim = rez.simScore  # this is the pairwise similarity score
    Nk = Xsim.shape[0]
    Xsim -= cp.diag(cp.diag(Xsim))

    # sort by firing rate first
    nspk = cp.zeros(Nk)
    for j in range(Nk):
        nspk[j] = cp.sum(rez.st3[:, 1] == (j + 1))  # determine total number of spikes in each neuron

    # (DEV_NOTES) currently neurons follow 1 based indexing, prior Python code may change this in which case the line above should be:
    #             nspk[j] = cp.sum(rez.st3[:, 1] == j)

    isort = cp.argsort(nspk)

    print('Initialised Spike Counts')

    if not flag:
        # if the flag is off, then no merges are performed
        # this function is then just used to compute cross- and auto- correlograms
        rez.R_CCG = cp.inf * cp.ones(Xsim.shape)
        rez.Q_CCG = cp.inf * cp.ones(Xsim.shape)
        rez.K_CCG = cp.zeros((*Xsim.shape, 2 * nbins + 1))

    for j in range(Nk):
        s1 = rez.st3[:, 0][rez.st3[:, 1] == (isort[j] + 1)] / ops.fs  # find all spikes from this cluster

        # (DEV_NOTES) above line will need modifying if neurons follow 0 based indexing and if rez.st3 ordering is reversed to optimise for C rather than Fortran

        if len(s1) != nspk[isort[j]]:
            print('Lost Track of Spike Counts')  # this is a check to make sure new clusters are combined correctly into bigger clusters

        # sort all the pairs of this neuron, discarding any that have fewer spikes

        # (DEV_NOTES) sorting below and calculating ienu can probably be improved

        ccsort = -cp.sort(-Xsim[isort[j]] * (nspk > len(s1)))  # sort in descending order
        ix = cp.argsort(-Xsim[isort[j]] * (nspk > len(s1)))

        if ccsort[len(ccsort) - 1] > 0.5:
            ienu = len(ccsort)
        else:
            ienu = cp.argmax(ccsort < 0.5)

        # for all pairs above 0.5 correlation

        for k in cp.arange(ienu):
            s2 = rez.st3[:, 0][rez.st3[:, 1] == (ix[k] + 1)] / ops.fs  # find the spikes of the pair

            # (DEV_NOTES) above line will need modifying if neurons follow 0 based indexing and if rez.st3 ordering is reversed to optimise for C rather than Fortran

            # compute cross-correlograms, refractoriness scores (Qi and rir), and normalization for these scores

            K, Qi, Q00, Q01, rir = ccg(s1, s2, nbins, dt)
            Q = cp.min(Qi / max(Q00, Q01))  # normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
            R = cp.min(rir)  # R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes

            if flag:
                if (Q < 0.2) and (R < 0.5):  # if both refractory criteria are met
                    i = ix[k]
                    # now merge j into i and move on
                    rez.st3[rez.st3[:, 1] == (isort[j] + 1), 1] = (i + 1)  # simply overwrite all the spikes of neuron j with i (i>j by construction)
                    nspk[i] += nspk[isort[j]]  # update number of spikes for cluster i
                    print(f'Merged {(isort[j] + 1)} into {(i + 1)}')
                    # YOU REALLY SHOULD MAKE SURE THE PC CHANNELS MATCH HERE
                    break
            else:
                # sometimes we just want to get the refractory scores and CCG
                rez.R_CCG[isort[j], ix[k]] = R
                rez.Q_CCG[isort[j], ix[k]] = Q

                rez.K_CCG[isort[j], ix[k]] = K
                rez.K_CCG[ix[k], isort[j]] = K[::-1]

    if not flag:
        rez.R_CCG = cp.minimum(rez.R_CCG, rez.R_CCG.T)  # symmetrize the scores
        rez.Q_CCG = cp.minimum(rez.Q_CCG, rez.Q_CCG.T)

    return rez


def set_cutoff(rez):
    # after everything else is done, this function takes spike trains and cuts off
    # any noise they might have picked up at low amplitude values
    # We look for bimodality in the amplitude plot, thus setting an individual threshold
    # for each neuron.
    # Also, this function calls "good" and "bad" clusters based on the auto-correlogram

    ops = rez.ops
    dt = 1 / 1000  # step size for CCG binning

    Nk = len(cp.unique(rez.st3[:, 1]))  # number of templates

    # (DEV_NOTES) easier way to calculate Nk using cp.max but this doesn't return an integer compatible with cp.zeros

    # sort by firing rate first

    rez.good = cp.zeros(Nk)

    rez.est_contam_rate = cp.zeros(Nk)

    for j in cp.arange(Nk):
        ix = cp.where(rez.st3[:, 1] == (j + 1))[0]
        ss = rez.st3[ix, 1] / ops.fs
        if len(ss) == 0:
            continue  # break if there are no spikes

        vexp = rez.st3[ix, 4]  # vexp is the relative residual variance of the spikes

        Th = ops.Th[0]  # start with a high threshold

        fcontamination = 0.1  # acceptable contamination rate

        while Th > ops.Th[1]:
            # continually lower the threshold, while the estimated unit contamination is low
            st = ss[vexp > Th]  # take spikes above the current threshold
            if len(st) == 0:
                Th -= 0.5  # if there are no spikes, we need to keep lowering the threshold
                continue

            K, Qi, Q00, Q01, rir = ccg(st, st, 500, dt)  # compute the auto-correlogram with 500 bins at 1ms bins
            Q = cp.min(Qi / max(Q00, Q01))  # this is a measure of refractoriness
            R = cp.min(rir)  # this is a second measure of refractoriness (kicks in for very low firing rates)
            if (Q > fcontamination) or (R > 0.05):  # if the unit is already contaminated, we break, and use the next higher threshold
                break
            else:
                if (Th == ops.Th[0]) and (Q < 0.05):
                    # only on the first iteration, we consider if the unit starts well isolated
                    # if it does, then we put much stricter criteria for isolation
                    # to make sure we don't settle for a relatively high contamination unit
                    fcontamination = min(0.05, max(0.01, Q * 2))

                    # if the unit starts out contaminated, we will settle with the higher contamination rate

                rez.good[j] = 1  # this unit is good, because we will stop lowering the threshold when it becomes bad
                Th -= 0.05

        Th += 0.05  # we exited the loop because the contamination was too high. We revert to the higher threshold
        st = ss[vexp > Th]
        K, Qi, Q00, Q01, rir = ccg(st, st, 500, dt)  # compute the auto-correlogram with 500 bins at 1ms bins
        Q = cp.min(Qi / max(Q00, Q01))  # this is a measure of refractoriness
        rez.est_contam_rate[j] = Q  # this score will be displayed in Phy

        rez.Ths[j] = Th  # store the threshold for potential debugging

        rez.st3[ix[vexp <= Th], 1] = 0  # any spikes below the threshold get discarded into a 0-th cluster

        # (DEV_NOTES) will need to change to -1-th cluster if spikes are 0-indexed

    # we sometimes get NaNs, why? replace with full contamination
    # (DEV_NOTES) this seems to occur when both Qi and max(Q00, Q01) are zero thus when dividing the
    #             two to get Q the result is a NaN

    rez.est_contam_rate[cp.isnan(rez.est_contam_rate)] = 1

    # remove spikes assigned to the 0 cluster

    # (DEV_NOTES) 0 cluster may change if clusters switch to 0-indexing

    ix = rez.st3[:, 1] == 0

    # (DEV_NOTES) "empty" values in code below needs checking before it can be used, in Matlab code
    #              [] is used for all cases, here I use cp.nan

    #     rez.st3[ix, :] = cp.nan

    #     if len(rez.cProj) != 0:
    #         rez.cProj[ix, :] = cp.nan # remove their template projections too
    #         rez.cProjPC(ix, :,:) = cp.nan # and their PC projections

    return rez


def splitAllClusters(rez, flag):
    # I call this algorithm "bimodal pursuit"
    # split clusters if they have bimodal projections
    # the strategy is to maximize a bimodality score and find a single vector projection
    # that maximizes it. If the distribution along that maximal projection crosses a
    # bimodality threshold, then the cluster is split along that direction
    # it only uses the PC features for each spike, stored in rez.cProjPC

    ops = rez.ops
    wPCA = ops.wPCA  # use PCA projections to reconstruct templates when we do splits

    ccsplit = rez.ops.AUCsplit  # this is the threshold for splits, and is one of the main parameters users can change

    NchanNear = min(ops.Nchan, 32)
    Nnearest = min(ops.Nchan, 32)
    sigmaMask = ops.sigmaMask

    ik = -1

    Nfilt = rez.W.shape[1] - 1

    nsplits = 0

    iC, mask, C2C = getClosestChannels(rez, sigmaMask, NchanNear)  # determine what channels each template lives on

    if 'nt0min' not in ops:
        ops.nt0min = 20         # the waveforms must be aligned to this sample

    iW = cp.argmax(cp.abs(rez.dWU[ops.nt0min, :, :]),
                   axis=1)  # find the peak abs channel for each template
    iW = iW.astype(cp.int32)

    isplit = cp.arange(1, Nfilt + 1)  # keep track of original cluster for each cluster. starts with all clusters being their own origin.

    # (DEV_NOTES) isplit will need modifying in 0-index case

    dt = 1 / 1000
    nccg = 0

    while ik < Nfilt:
        if ik % 100 == 1:
            # periodically write updates
            print(f'Found {nsplits} splits, checked {ik + 1}/{Nfilt + 1} clusters, nccg {nccg}')
        ik += 1

        isp = cp.where(rez.st3[:, 1] == (ik + 1))  # get all spikes from this cluster
        nSpikes = len(isp)

        if nSpikes < 300:
            continue  # do not split if fewer than 300 spikes (we cannot estimate cross-correlograms accurately)

        ss = rez.st3[:, 1][isp] / ops.fs  # convert to seconds

        clp0 = rez.cProjPC[isp, :, :]  # get the PC projections for these spikes
        clp0 = cp.array(clp0.reshape(clp0.shape[0], -1))
        clp = clp0 - cp.mean(clp0, axis=0)  # mean center them

        # (DEV_NOTES) Python flattens clp0 in C order rather than Fortran order so the flattened PC projections will be slightly different,
        #             however this is fixed when the projections are reformed later

        clp -= my_conv2(clp, 250, 1)  # subtract a running average, because the projections are NOT drift corrected

        # now use two different ways to initialize the bimodal direction
        # the main script calls this function twice, and does both initializations

        if flag:
            u, s, v = svdecon(clp.T)
            w = u[:, 0]  # initialize with the top PC
        else:
            w = cp.mean(clp0, axis=0)  # initialize with the mean of NOT drift-corrected trace
            w /= cp.sum(w ** 2) ** 0.5  # unit-normalize

        # initial projections of waveform PCs onto 1D vector

        x = cp.dot(clp, w)
        s1 = cp.var(x[x > cp.mean(x)])  # initialize estimates of variance for the first
        s2 = cp.var(x[x < cp.mean(x)])  # and second gaussian in the mixture of 1D gaussians

        mu1 = cp.mean(x[x > cp.mean(x)])  # initialize the means as well
        mu2 = cp.mean(x[x < cp.mean(x)])
        p = cp.mean(x > cp.mean(x))  # and the probability that a spike is assigned to the first Gaussian

        logp = cp.zeros((len(isp), 2))  # initialize matrix of log probabilities that each spike is assigned to the first or second cluster

        # do 50 pursuit iteration

        logP = cp.zeros(50)  # used to monitor the cost function

        for k in range(50):
            # for each spike, estimate its probability to come from either Gaussian cluster
            logp[:, 0] = -1 / 2 * log(s1) - ((x - mu1) ** 2) / (2 * s1) + log(p)
            logp[:, 1] = -1 / 2 * log(s2) - ((x - mu2) ** 2) / (2 * s2) + log(1 - p)

            lMax = cp.max(logp, axis=1)
            logp -= lMax[:, cp.newaxis]  # subtract the max for floating point accuracy
            rs = cp.exp(logp)  # exponentiate the probabilities

            pval = cp.log(cp.sum(rs, axis=1)) + lMax  # get the normalizer and add back the max
            logP[k] = cp.mean(pval)  # this is the cost function: we can monitor its increase

            rs /= cp.sum(rs, axis=1)[:, cp.newaxis]  # normalize so that probabilities sum to 1

            p = cp.mean(rs[:, 0])  # mean probability to be assigned to Gaussian 1
            mu1 = cp.dot(rs[:, 0], x) / cp.sum(rs[:, 0])  # new estimate of mean of cluster 1 (weighted by "responsibilities")
            mu2 = cp.dot(rs[:, 1], x) / cp.sum(rs[:, 1])  # new estimate of mean of cluster 2 (weighted by "responsibilities")

            s1 = cp.dot(rs[:, 0], (x - mu1) ** 2) / cp.sum(rs[:, 0])  # new estimates of variances
            s2 = cp.dot(rs[:, 1], (x - mu2) ** 2) / cp.sum(rs[:, 1])

            if (k > 10) and (k % 2 == 1):
                # starting at iteration 10, we start re-estimating the pursuit direction
                # that is, given the Gaussian cluster assignments, and the mean and variances,
                # we re-estimate w
                StS = cp.matmul(clp.T, clp * (rs[:, 0] / s1 + rs[:, 1] / s2)[:,
                                             cp.newaxis]) / nSpikes  # these equations follow from the model
                StMu = cp.dot(clp.T, rs[:, 0] * mu1 / s1 + rs[:, 1] * mu2 / s2) / nSpikes

                w = cp.linalg.solve(StS.T, StMu)  # this is the new estimate of the best pursuit direction
                w /= cp.sum(w ** 2) ** 0.5  # which we unit normalize
                x = cp.dot(clp, w)

        ilow = rs[:, 0] > rs[:, 1]  # these spikes are assigned to cluster 1
        plow = cp.mean(rs[:, 0][ilow])  # the mean probability of spikes assigned to cluster 1
        phigh = cp.mean(rs[:, 1][~ilow])  # same for cluster 2
        nremove = min(cp.mean(ilow), cp.mean(~ilow))  # the smallest cluster has this proportion of all spikes

        # did this split fix the autocorrelograms?
        K, Qi, Q00, Q01, rir = ccg(ss[ilow], ss[~ilow], 500, dt)  # compute the cross-correlogram between spikes in the putative new clusters
        Q12 = cp.min(Qi / max(Q00, Q01))  # refractoriness metric 1
        R = cp.min(rir)  # refractoriness metric 2

        # if the CCG has a dip, don't do the split.
        # These thresholds are consistent with the ones from merges.
        if (Q12 < 0.25) and (R < 0.05):  # if both metrics are below threshold.
            nccg += 1  # keep track of how many splits were voided by the CCG criterion
            continue

        # now decide if the split would result in waveforms that are too similar
        c1 = cp.matmul(wPCA, cp.reshape(cp.mean(clp0[ilow, :], 0), 3, -1))  # the reconstructed mean waveforms for putative cluster 1
        c2 = cp.matmul(wPCA, cp.reshape(cp.mean(clp0[~ilow, :], 0), 3, -1))  # the reconstructed mean waveforms for putative cluster 2
        cc = cp.corrcoef(c1, c2)  # correlation of mean waveforms
        n1 = sqrt(cp.sum(c1 ** 2))  # the amplitude estimate 1
        n2 = sqrt(cp.sum(c2 ** 2))  # the amplitude estimate 2

        r0 = 2 * abs((n1 - n2) / (n1 + n2))

        # if the templates are correlated, and their amplitudes are similar, stop the split!!!

        if (cc[0, 1] > 0.9) and (r0 < 0.2):
            continue

        # finaly criteria to continue with the split: if the split piece is more than 5% of all spikes,
        # if the split piece is more than 300 spikes, and if the confidences for assigning spikes to
        # both clusters exceeds a preset criterion ccsplit
        if (nremove > 0.05) and (min(plow, phigh) > ccsplit) and (min(cp.sum(ilow), cp.sum(~ilow)) > 300):
            # one cluster stays, one goes
            Nfilt += 1

            # the templates for the splits have been estimated from PC coefficients

            # (DEV_NOTES) code below involves multiple CuPy arrays changing shape to accomodate the extra cluster, this could potentially
            #             be done more efficiently?

            rez.dWU = cp.concatenate((rez.dWU, cp.zeros((*rez.dWU.shape[:-1], 1))), axis=2)
            rez.dWU[:, iC[:, iW[ik]], Nfilt] = c2
            rez.dWU[:, iC[:, iW[ik]], ik] = c1

            # the temporal components are therefore just the PC waveforms
            rez.W = cp.concatenate((rez.W, wPCA.reshape(wPCA.shape[0], 1, wPCA.shape[1])), axis=1)
            iW = cp.concatenate((iW, iW[ik]))  # copy the best channel from the original template
            isplit = cp.concatenate((isplit, isplit[ik]))  # copy the provenance index to keep track of splits

            rez.st3[isp[ilow], 1] = (Nfilt + 1)  # overwrite spike indices with the new index
            rez.simScore = cp.concatenate((rez.simScore, rez.simScore[:, ik].reshape(-1, 1)), axis=1)  # copy similarity scores from the original
            rez.simScore = cp.concatenate((rez.simScore, rez.simScore[ik, :].reshape(1, -1)), axis=0)  # copy similarity scores from the original
            rez.simScore[ik, Nfilt] = 1  # set the similarity with original to 1
            rez.simScore[Nfilt, ik] = 1  # set the similarity with original to 1

            rez.iNeigh = cp.concatenate((rez.iNeigh, rez.iNeigh[:, ik].reshape(-1, 1)), axis=0)  # copy neighbor template list from the original
            rez.iNeighPC = cp.concatenate((rez.iNeighPC, rez.iNeighPC[:, ik].reshape(-1, 1)), axis=0)  # copy neighbor channel list from the original

            # try this cluster again
            ik -= 1  # the cluster piece that stays at this index needs to be tested for splits again before proceeding
            # the piece that became a new cluster will be tested again when we get to the end of the list
            nsplits += 1  # keep track of how many splits we did

    print(f'Finished splitting. Found {nsplits} splits, checked {ik + 1}/{Nfilt + 1} clusters, nccg {nccg}')

    Nfilt = rez.W.shape[1]  # new number of templates
    Nrank = 3
    Nchan = ops.Nchan
    Params = cp.array([0, Nfilt, 0, 0, rez.W.shape[0], Nnearest, Nrank, 0, 0, Nchan, NchanNear,
                       ops.nt0min, 0], dtype=cp.float64)  # make a new Params to pass on parameters to CUDA

    # we need to re-estimate the spatial profiles

    # (DEV_NOTES) line below is incompatible with new implementation of getKernels

    Ka, Kb = getKernels(Bunch({'nt0min':ops.nt0min, 'nt0':ops.nt0, 'nup':10, 'sig':1}))  # we get the time upsampling kernels again
    rez.W, rez.U, rez.mu = mexSVDsmall2(Params, rez.dWU, rez.W, iC - 1, iW - 1, Ka, Kb)  # we run SVD

    WtW, iList = getMeWtW(rez.W.astype(cp.float32), rez.U.astype(cp.float32), Nnearest)  # we re-compute similarity scores between templates
    rez.iList = iList  # over-write the list of nearest templates

    isplit = rez.simScore == 1  # overwrite the similarity scores of clusters with same parent
    rez.simScore = cp.max(WtW, axis=2)
    rez.simScore[isplit] = 1  # 1 means they come from the same parent

    rez.iNeigh = iList[:, 0:Nfilt]  # get the new neighbor templates
    rez.iNeighPC = iC[:, iW[0:Nfilt]]  # get the new neighbor channels

    # for Phy, we need to pad the spikes with zeros so the spikes are aligned to the center of the window
    rez.Wphy = cp.concatenate((cp.zeros((1 + ops.ntomin, Nfilt + 1, Nrank)), rez.W), axis=0)

    rez.isplit = isplit  # keep track of origins for each cluster
