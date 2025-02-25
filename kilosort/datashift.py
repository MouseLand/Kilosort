import logging
logger = logging.getLogger(__name__)

from scipy.sparse import coo_matrix
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from kilosort import spikedetect


def bin_spikes(ops, st):
    """ for each batch, the spikes in that batch are binned to a 2D matrix by amplitude and depth
    """

    # the bin edges are based on min and max of channel y positions
    ymin = ops['yc'].min()
    ymax = ops['yc'].max()
    dd = ops['binning_depth'] # binning width in depth
    
    # start 1um below the lowest channel
    dmin = ymin-1

    # dmax is how many bins to use
    dmax = 1 + np.ceil((ymax-dmin)/dd).astype('int32')

    Nbatches = ops['Nbatches']
    
    batch_id = st[:,4].copy()

    # always use 20 bins for amplitude binning
    F = np.zeros((Nbatches, dmax, 20))
    for t in range(ops['Nbatches']):
        # consider only spikes from this batch
        ix = (batch_id==t).nonzero()[0]
        sst = st[ix]

        # their depth relative to the minimum
        dep = sst[:,1] - dmin

        # the amplitude binnning is logarithmic, goes from the Th_universal minimum value to 100. 
        amp = np.log10(np.minimum(99, sst[:,2])) - np.log10(ops['Th_universal'])

        # amplitudes get normalized from 0 to 1
        amp = amp / (np.log10(100)-np.log10(ops['Th_universal']))

        # rows are divided by the vertical binning depth
        rows = (dep/dd).astype('int32')

        # columns are from 0 to 20
        cols = (1e-5 + amp * 20).astype('int32')

        # for efficient binning, use sparse matrix computation in scipy
        cou = np.ones(len(ix))
        M = coo_matrix((cou, (rows, cols)), (dmax, 20))

        # the 2D histogram counts are transformed to logarithm
        F[t] = np.log2(1+M.todense())

    # center of each vertical sampling bin
    ysamp = dmin + dd * np.arange(dmax) - dd/2

    return F, ysamp


def align_block2(F, ysamp, ops, device=torch.device('cuda')):

    Nbatches = ops['Nbatches']
    
    # n is the maximum vertical shift allowed, in units of bins
    n = 15
    dc = np.zeros((2*n+1, Nbatches))
    dt = np.arange(-n,n+1,1)

    # batch fingerprints are mean subtracted along depth
    Fg = torch.from_numpy(F).to(device).float() 
    Fg = Fg - Fg.mean(1).unsqueeze(1)

    # the template fingerprint is initialized with batch 300 if that exists
    F0 = Fg[np.minimum(300, Nbatches//2)]

    niter = 10
    dall = np.zeros((niter, Nbatches))

    # at each iteration, align each batch to the template fingerprint
    # Fg is incrementally modified, and cumulative shifts are accumulated over iterations
    for iter in range(niter):
        # for each vertical shift in the range -n to n, compute the dot product
        for t in range(len(dt)):
            Fs = torch.roll(Fg, dt[t], 1)
            dc[t] = (Fs * F0).mean(-1).mean(-1).cpu().numpy()

        # for all but the last iteration, align the batches 
        if iter<niter-1:
            # the maximum dot product is the best match for each batch
            imax = np.argmax(dc, 0)

            for t in range(len(dt)):
                # for batches which have the maximum at dt[t]
                ib = imax==t

                # roll the fingerprints for those batches by dt[t]
                Fg[ib] = torch.roll(Fg[ib], dt[t], 1)
                dall[iter, ib] = dt[t]

        # take the mean of the aligned batches. This will be the new fingerprint template. 
        F0 = Fg.mean(0)


    # divide the vertical bins into nblocks non-overlapping segments, and then consider also the segments which half-overlap these segments 
    nblocks = ops['nblocks']
    nybins = F.shape[1]
    yl = nybins//nblocks
    ifirst = np.round(np.linspace(0,nybins-yl, 2 *nblocks-1)).astype('int32')
    ilast = ifirst + yl

    # the new nblocks is 2*nblocks - 1 due to the overlapping blocks
    nblocks = len(ifirst)
    yblk = np.zeros(nblocks,)
    
    # consider much smaller ranges for the fine drift correction
    n  = 5
    dt = np.arange(-n, n+1, 1)
    dcs = np.zeros((2*n+1, Nbatches, nblocks))

    # for each block in each batch, recompute the dot products with the template
    for j in range(nblocks):
        isub = np.arange(ifirst[j], ilast[j], 1)
        yblk[j] = ysamp[isub].mean()

        Fsub = Fg[:, isub]

        for t in range(len(dt)):
            Fs = torch.roll(Fsub, dt[t], 1)
            dcs[t, :, j] = (Fs * F0[isub]).mean(-1).mean(-1).cpu().numpy()

    # upsamples the dot-product matrices by 10 to get finer estimates of vertica ldrift
    dtup = np.linspace(-n,n,2*n*10+1)

    # get 1D upsampling matrix
    Kn = kernelD(dt,dtup,1) 

    # smooth the dot-product matrices across correlation, batches, and vertical offsets
    dcs = gaussian_filter(dcs, ops['drift_smoothing'])

    # for each block, upsample the dot-product matrix and find new max
    imin = np.zeros((Nbatches, nblocks))
    for j in range(nblocks):
        dcup = Kn.T @ dcs[:,:,j]
        imax = np.argmax(dcup, 0)

        # the new max gets added to the last iteration of dall
        dall[niter-1] = dtup[imax]

        # the cumulative shifts in dall represent the total vertical shift for each batch
        imin[:,j] = dall.sum(0)

    # Fg gets reinitialized with the un-corrected F without subtracting the mean across depth.      
    Fg = torch.from_numpy(F).float()
    imax = dall[:niter-1].sum(0)

    # Fg gets aligned again to compute the non-mean subtracted fingerprint    
    for t in range(len(dt)):
        ib = imax==dt[t]
        Fg[ib] = torch.roll(Fg[ib], dt[t], 1)
    F0m = Fg.mean(0)

    return imin, yblk, F0, F0m


def kernelD(x, y, sig = 1):    
    ds = (x[:,np.newaxis] - y)
    Kn = np.exp(-ds**2 / (2*sig**2))
    return Kn
    
def kernel2D_torch(x, y, sig = 1):    
    ds = ((x.unsqueeze(1) - y)**2).sum(-1)
    Kn = torch.exp(-ds / (2*sig**2))
    return Kn

def kernel2D(x, y, sig = 1):
    ds = ((x[:,np.newaxis] - y)**2).sum(-1)
    Kn = np.exp(-ds / (2*sig**2))
    return Kn

def run(ops, bfile, device=torch.device('cuda'), progress_bar=None,
        clear_cache=False, verbose=False):
    """ this step computes a drift correction model
    it returns vertical correction amplitudes for each batch, and for multiple blocks in a batch if nblocks > 1. 
    """
    
    if ops['nblocks']<1:
        ops['dshift'] = None 
        logger.info('nblocks = 0, skipping drift correction')
        return ops, None
    
    # the first step is to extract all spikes using the universal templates
    st, _, ops  = spikedetect.run(
        ops, bfile, device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )

    # spikes are binned by amplitude and y-position to construct a "fingerprint" for each batch
    F, ysamp = bin_spikes(ops, st)

    # the fingerprints are iteratively aligned to each other vertically
    imin, yblk, _, _ = align_block2(F, ysamp, ops, device=device)

    # imin contains the shifts for each batch, in units of discrete bins
    # multiply back with binning_depth for microns
    dshift = imin * ops['binning_depth']

    # we save the variables needed for drift correction during the data preprocessing step
    ops['yblk'] = yblk
    ops['dshift'] = dshift 
    xp = np.vstack((ops['xc'],ops['yc'])).T

    # for interpolation, we precompute a radial kernel based on distances between sites
    Kxx = kernel2D(xp, xp, ops['sig_interp'])
    Kxx = torch.from_numpy(Kxx).to(device)

    # a small constant is added to the diagonal for stability of the matrix inversion
    ops['iKxx'] = torch.linalg.inv(Kxx + 0.01 * torch.eye(Kxx.shape[0], device=device))

    return ops, st
