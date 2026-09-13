import logging
logger = logging.getLogger(__name__)

import warnings

from scipy.sparse import coo_matrix
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from kilosort import spikedetect


def bin_spikes(ops, st, group_id=None, n_groups=None):
    """ the spikes in each group are binned to a 2D matrix by amplitude and depth

    By default each batch is its own group, so this returns one "fingerprint"
    per batch. Passing `group_id` allows spikes to be pooled over larger units
    of time instead, which is used by the chronic drift mode to build one
    fingerprint per recording segment (e.g. per day).

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    st : np.ndarray
        Spike times variable from `spikedetect.run`, with batch index in
        column 4.
    group_id : np.ndarray; optional.
        Integer group index for each spike, which must be sorted in
        non-decreasing order. Defaults to the batch index, `st[:,4]`.
    n_groups : int; optional.
        Number of groups. Defaults to `ops['Nbatches']`.

    Returns
    -------
    F : np.ndarray
        Fingerprints with shape `(n_groups, dmax, 20)`.
    ysamp : np.ndarray
        Center of each vertical sampling bin, with shape `(dmax,)`.

    """

    # the bin edges are based on min and max of channel y positions
    ymin = ops['yc'].min()
    ymax = ops['yc'].max()
    dd = ops['binning_depth'] # binning width in depth

    # start 1um below the lowest channel
    dmin = ymin-1

    # dmax is how many bins to use
    dmax = 1 + np.ceil((ymax-dmin)/dd).astype('int32')

    if group_id is None:
        # one group per batch, the default
        group_id = st[:,4]
        n_groups = ops['Nbatches']
    group_id = np.asarray(group_id).astype('int64')
    if group_id.size > 0 and np.any(np.diff(group_id) < 0):
        raise ValueError('`group_id` must be sorted in non-decreasing order.')

    # spikes are detected batch by batch in ascending order, so group ids are
    # sorted and the spikes belonging to each group are a contiguous slice.
    # Looking the slices up with a binary search avoids rescanning the full
    # spike array once per group, which matters for long recordings.
    edges = np.searchsorted(group_id, np.arange(n_groups+1), side='left')

    # always use 20 bins for amplitude binning
    F = np.zeros((n_groups, dmax, 20))
    for t in range(n_groups):
        # consider only spikes from this group
        sst = st[edges[t]:edges[t+1]]
        if sst.shape[0] == 0:
            continue

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
        cou = np.ones(sst.shape[0])
        M = coo_matrix((cou, (rows, cols)), (dmax, 20))

        # the 2D histogram counts are transformed to logarithm
        F[t] = np.log2(1+M.todense())

    # center of each vertical sampling bin
    ysamp = dmin + dd * np.arange(dmax) - dd/2

    return F, ysamp


def block_indices(nybins, nblocks):
    """Vertical bin ranges for each registration block.

    The depth axis is divided into `nblocks` non-overlapping segments, plus the
    `nblocks-1` segments which half-overlap those, giving `2*nblocks-1` blocks.

    """
    yl = nybins//nblocks
    ifirst = np.round(np.linspace(0, nybins-yl, 2*nblocks-1)).astype('int32')
    ilast = ifirst + yl

    return ifirst, ilast


def align_block2(F, ysamp, ops, device=torch.device('cuda'),
                 drift_smoothing=None):
    """Register fingerprints to each other along the depth axis.

    `F` holds one fingerprint per registration unit: per batch by default, or
    per recording segment when running in chronic drift mode. The returned
    shifts have one row per fingerprint, so the caller decides what a row means.

    """

    # number of registration units, which is not necessarily ops['Nbatches']
    Nbatches = F.shape[0]
    if Nbatches < 2:
        raise ValueError('Drift registration needs at least 2 fingerprints, '
                         f'but got {Nbatches}.')

    if drift_smoothing is None:
        drift_smoothing = ops['drift_smoothing']

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
    ifirst, ilast = block_indices(F.shape[1], ops['nblocks'])

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
    dcs = gaussian_filter(dcs, drift_smoothing)

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


def segment_batches(ops, bfile):
    """Assign each batch to a recording segment, for chronic drift correction.

    A batch belongs to the segment which contains its first sample. Batches
    which span a boundary mix data from two segments, so they are flagged and
    excluded when pooling spikes into fingerprints. They still receive a shift,
    that of the segment they start in.

    Segments which contain no data (because `tmin` / `tmax` cropped them away)
    are dropped, and the remaining ones are relabeled from 0.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
        Must contain 'drift_segment_starts', the start sample of each segment.
    bfile : kilosort.io.BinaryFiltered
        Wrapped file object for handling data.

    Returns
    -------
    seg_of_batch : np.ndarray
        Segment index for each batch, with shape `(Nbatches,)`.
    straddles : np.ndarray
        Boolean mask of batches which span a segment boundary, with shape
        `(Nbatches,)`.
    n_segments : int
        Number of segments which contain data.

    """

    starts = np.asarray(ops['drift_segment_starts'], dtype='int64')

    # batch b covers raw samples [imin + b*NT, imin + (b+1)*NT), where NT
    # includes batch_downsampling since padded_batch_to_torch multiplies the
    # batch index by it before looking up sample indices
    NT = np.int64(bfile.NT) * np.int64(bfile.batch_downsampling)
    imin = np.int64(bfile.imin)
    nb = np.int64(bfile.n_batches)

    first = imin + np.arange(nb, dtype='int64') * NT
    last = first + NT - 1

    seg_first = np.searchsorted(starts, first, side='right') - 1
    seg_last = np.searchsorted(starts, last, side='right') - 1
    straddles = seg_first != seg_last

    # drop segments with no data and relabel the rest to 0...n_segments-1
    used, seg_of_batch = np.unique(seg_first, return_inverse=True)
    n_segments = used.size
    ops['drift_segment_used'] = used
    ops['drift_segment_starts_used'] = starts[used]

    if n_segments < 2:
        raise ValueError(
            f'Chronic drift correction needs at least 2 segments containing '
            f'data, but only {n_segments} of {starts.size} segment(s) fall '
            'within [tmin, tmax]. Check `drift_segment_starts`, `tmin` and '
            '`tmax`.'
            )

    counts = np.bincount(seg_of_batch, minlength=n_segments)
    logger.info(f'Chronic drift mode: {n_segments} segments, '
                f'{counts.min()}-{counts.max()} batches each. '
                f'{straddles.sum()} batches span a segment boundary and are '
                'excluded from the fingerprints.')

    return seg_of_batch.astype('int64'), straddles, n_segments


def segment_residual_drift(ops, st, seg_of_batch, F_seg, straddles=None,
                           device=torch.device('cuda'), max_batches=2000):
    """Estimate the residual per-batch shift within each segment.

    This is a diagnostic: it does not change `dshift`. Chronic drift mode
    assumes the shift is constant within a segment, so every batch of a segment
    should register to that segment's own pooled fingerprint at roughly zero
    offset. A large spread means there is real within-segment drift; a
    consistently nonzero median means the pooled registration for that segment
    is biased, which usually indicates the fingerprint changed in content
    rather than position.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    st : np.ndarray
        Spike times variable from `spikedetect.run`.
    seg_of_batch : np.ndarray
        Segment index for each batch, from `segment_batches`.
    F_seg : np.ndarray
        Pooled fingerprint per segment, with shape `(n_segments, dmax, 20)`.
    straddles : np.ndarray; optional.
        Boolean mask of batches which span a segment boundary, from
        `segment_batches`. Those batches contain data from two segments, so
        their residual is meaningless and they are skipped.
    device : torch.device
        Indicates whether `pytorch` operations should be run on cpu or gpu.
    max_batches : int; default=2000.
        Maximum number of batches sampled per segment, to bound memory use.

    Returns
    -------
    ops : dict
        With 'drift_residual', 'drift_residual_batches' and
        'drift_residual_summary' added.

    """

    n_segments = F_seg.shape[0]
    dd = ops['binning_depth']
    ifirst, ilast = block_indices(F_seg.shape[1], ops['nblocks'])
    nblocks = len(ifirst)

    # same fine search range and upsampling as the block step of align_block2
    n = 5
    dt = np.arange(-n, n+1, 1)
    dtup = np.linspace(-n, n, 2*n*10+1)
    Kn = kernelD(dt, dtup, 1)

    sp_batch = st[:,4].astype('int64')
    F0_all = torch.from_numpy(F_seg).to(device).float()
    F0_all = F0_all - F0_all.mean(1).unsqueeze(1)

    residual = []
    batches = []
    measured = []
    summary = np.full((n_segments, nblocks, 4), np.nan)

    usable = np.ones(seg_of_batch.size, dtype='bool') if straddles is None \
             else ~np.asarray(straddles)

    for s in range(n_segments):
        ib = np.flatnonzero((seg_of_batch == s) & usable)
        if ib.size == 0:
            continue
        if ib.size > max_batches:
            # subsample evenly so that cost and memory stay bounded
            ib = ib[::int(np.ceil(ib.size / max_batches))]

        # per-batch fingerprints for this segment only, relabeled to 0...k-1.
        # Batch ids are sorted, so the segment's spikes are a contiguous slice.
        lo = np.searchsorted(sp_batch, ib[0], side='left')
        hi = np.searchsorted(sp_batch, ib[-1], side='right')
        keep = np.isin(sp_batch[lo:hi], ib)
        local = np.searchsorted(ib, sp_batch[lo:hi][keep])
        Fb, _ = bin_spikes(ops, st[lo:hi][keep], group_id=local,
                           n_groups=ib.size)

        Fb = torch.from_numpy(Fb).to(device).float()
        Fb = Fb - Fb.mean(1).unsqueeze(1)
        F0 = F0_all[s]

        res = np.zeros((ib.size, nblocks))
        for j in range(nblocks):
            isub = np.arange(ifirst[j], ilast[j], 1)
            Fsub = Fb[:, isub]
            dcs = np.zeros((len(dt), ib.size))
            for t in range(len(dt)):
                Fs = torch.roll(Fsub, dt[t], 1)
                dcs[t] = (Fs * F0[isub]).mean(-1).mean(-1).cpu().numpy()
            res[:,j] = dtup[np.argmax(Kn.T @ dcs, 0)] * dd
            summary[s,j] = [np.median(res[:,j]), res[:,j].std(),
                            np.percentile(res[:,j], 5),
                            np.percentile(res[:,j], 95)]

        residual.append(res)
        batches.append(ib)
        measured.append(s)

    residual = np.concatenate(residual, axis=0)
    batches = np.concatenate(batches, axis=0)
    ops['drift_residual'] = residual
    ops['drift_residual_batches'] = batches
    ops['drift_residual_summary'] = summary

    # report the worst block of each segment
    tol = max(dd, 0.5*ops['sig_interp'])
    spread = summary[:,:,3] - summary[:,:,2]
    logger.info(' ')
    logger.info('Within-segment residual drift (constant-shift assumption):')
    logger.info(f'{"seg":>4} {"batches":>8} {"median":>9} {"p5..p95":>18} '
                f'{"max|res|":>9}   (um)')
    for i, s in enumerate(measured):
        j = int(np.argmax(spread[s]))
        res_s = residual[i]
        logger.info(
            f'{s:>4} {res_s.shape[0]:>8} {summary[s,j,0]:>+9.2f} '
            f'{summary[s,j,2]:>+8.2f}..{summary[s,j,3]:>+8.2f} '
            f'{np.abs(res_s).max():>9.2f}'
            )

    bad_spread = np.flatnonzero(np.nanmax(spread, axis=1) > tol)
    if bad_spread.size > 0:
        warnings.warn(
            f'Within-segment drift exceeds {tol:.1f} um (p5-p95) for '
            f'segment(s) {bad_spread.tolist()}. The assumption that drift is '
            'constant within a segment may not hold for these. Consider '
            'splitting them into shorter segments, or using standard '
            'per-batch drift correction (`drift_segment_starts=None`).',
            UserWarning
            )

    bad_bias = np.flatnonzero(np.nanmax(np.abs(summary[:,:,0]), axis=1) > tol)
    if bad_bias.size > 0:
        warnings.warn(
            f'Median within-segment residual exceeds {tol:.1f} um for '
            f'segment(s) {bad_bias.tolist()}. Their pooled registration may be '
            'biased, which usually means the spike depth/amplitude '
            'distribution changed in content rather than only in position '
            '(e.g. units lost or gained, or a gain change).',
            UserWarning
            )

    return ops


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

    if ops.get('drift_segment_starts', None) is None:
        # spikes are binned by amplitude and y-position to construct a "fingerprint" for each batch
        F, ysamp = bin_spikes(ops, st)

        # the fingerprints are iteratively aligned to each other vertically
        imin, yblk, _, _ = align_block2(F, ysamp, ops, device=device)
        ops['batch_to_segment'] = None
    else:
        # chronic mode: the shift is held constant within each segment, so all
        # of a segment's spikes are pooled into a single fingerprint and only
        # one shift per (segment, block) is estimated
        seg_of_batch, straddles, n_segments = segment_batches(ops, bfile)
        sp_batch = st[:,4].astype('int64')

        # batches spanning a boundary contain data from two segments
        keep = ~straddles[sp_batch]
        group_id = seg_of_batch[sp_batch[keep]]
        F, ysamp = bin_spikes(ops, st[keep], group_id=group_id,
                              n_groups=n_segments)

        # never smooth the dot products across segments: a shift between two
        # recording days is a real step, not noise. The smoothing axes are
        # (correlation, registration unit, depth block).
        smoothing = list(ops['drift_smoothing'])
        smoothing[1] = 0.0

        imin_seg, yblk, _, _ = align_block2(F, ysamp, ops, device=device,
                                            drift_smoothing=smoothing)

        # broadcast each segment's shift back to all of its batches, so that
        # dshift keeps its usual (Nbatches, nblocks) shape downstream
        imin = imin_seg[seg_of_batch]
        ops['batch_to_segment'] = seg_of_batch
        ops['drift_segment_shift'] = imin_seg * ops['binning_depth']

        if ops.get('drift_segment_diagnostics', True):
            ops = segment_residual_drift(ops, st, seg_of_batch, F,
                                         straddles=straddles, device=device)

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
