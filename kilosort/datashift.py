import logging
logger = logging.getLogger(__name__)

import warnings

from scipy.sparse import coo_matrix
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from kilosort import spikedetect


def bin_spikes(ops, st, group_id=None, n_groups=None):
    """Bin spikes into (depth x log-amplitude) fingerprints, one per group.

    By default each batch forms its own group, so that every batch gets its own
    2D histogram ("fingerprint") of spikes over depth and log-amplitude. For
    chronic drift correction, `group_id` is instead the recording segment each
    spike belongs to, which pools a whole segment into a single fingerprint.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    st : np.ndarray
        Spike times array with 6 columns, as returned by `spikedetect.run`.
    group_id : np.ndarray; optional.
        Integer group index for each spike, **sorted non-decreasing**. Defaults
        to the batch index, `st[:,4]`, reproducing the original behavior.
    n_groups : int; optional.
        Number of groups. Defaults to `ops['Nbatches']`.

    Returns
    -------
    F : np.ndarray
        Fingerprints, shape (n_groups, dmax, 20).
    ysamp : np.ndarray
        Center of each vertical sampling bin, shape (dmax,).

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
        group_id = st[:,4]
        n_groups = ops['Nbatches']
    group_id = np.asarray(group_id).astype('int64')
    if group_id.size > 0 and np.any(np.diff(group_id) < 0):
        raise ValueError('bin_spikes requires `group_id` sorted non-decreasing')

    # Group boundaries in O(n_groups * log n_spikes). The previous
    # implementation rescanned the full spike array once per group, which is
    # O(n_groups * n_spikes) and becomes the dominant cost for long recordings.
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
    """Get the first and last vertical bin of each registration block.

    Depth is divided into `nblocks` non-overlapping segments, plus the
    `nblocks - 1` segments which half-overlap those, for `2*nblocks - 1` blocks
    in total.

    Parameters
    ----------
    nybins : int
        Number of vertical bins in the fingerprints (`F.shape[1]`).
    nblocks : int
        Number of non-overlapping blocks, i.e. `ops['nblocks']`.

    Returns
    -------
    ifirst : np.ndarray
        First vertical bin of each block, shape (2*nblocks - 1,).
    ilast : np.ndarray
        One past the last vertical bin of each block, shape (2*nblocks - 1,).

    """

    yl = nybins//nblocks
    ifirst = np.round(np.linspace(0, nybins-yl, 2*nblocks-1)).astype('int32')
    ilast = ifirst + yl

    return ifirst, ilast


def align_block2(F, ysamp, ops, device=torch.device('cuda'),
                 drift_smoothing=None):
    """Register fingerprints to each other to estimate vertical drift.

    Parameters
    ----------
    F : np.ndarray
        Fingerprints from `bin_spikes`, shape (n, dmax, 20). Note that `n` is
        the number of registration units, which is the number of batches for
        standard drift correction and the number of segments in chronic mode.
    ysamp : np.ndarray
        Center of each vertical sampling bin.
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    device : torch.device; default=torch.device('cuda').
    drift_smoothing : list of float; optional.
        Overrides `ops['drift_smoothing']`. Used by chronic mode to disable
        smoothing along the time axis, since a segment-to-segment step is real
        signal rather than estimation noise.

    """

    if F.shape[0] < 2:
        raise ValueError('Drift registration needs at least 2 fingerprints; '
                         f'got {F.shape[0]}.')

    # number of registration units (batches, or segments in chronic mode)
    Nbatches = F.shape[0]

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
    ifirst, ilast = block_indices(nybins, nblocks)

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
    if drift_smoothing is None:
        drift_smoothing = ops['drift_smoothing']
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
    """Assign each sorted batch to a recording segment.

    A batch belongs to the segment containing its *first* sample. Batches that
    span a boundary are flagged; they mix data from two segments and are
    excluded from fingerprint pooling, but they still receive a shift.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
        Must contain `drift_segment_starts`.
    bfile : kilosort.io.BinaryFiltered
        Wrapped file object for handling data.

    Returns
    -------
    seg_of_batch : np.ndarray
        Segment index of each batch, shape (Nbatches,), with values in
        [0, n_segments).
    straddles : np.ndarray
        Boolean mask of batches which span a segment boundary, shape
        (Nbatches,).
    n_segments : int
        Number of segments which contain data.

    """

    starts = np.asarray(ops['drift_segment_starts'], dtype='int64')

    # Batch `b` (sorted index) covers raw samples
    #   [imin + b*NT*ds, imin + (b+1)*NT*ds)
    # see io.BinaryRWFile._get_batch_edges and padded_batch_to_torch, which does
    # `ibatch *= self.batch_downsampling`.
    NT = np.int64(bfile.NT) * np.int64(bfile.batch_downsampling)
    imin = np.int64(bfile.imin)
    nb = np.int64(bfile.n_batches)

    first = imin + np.arange(nb, dtype='int64') * NT
    last = first + NT - 1

    seg_first = np.searchsorted(starts, first, side='right') - 1
    seg_last = np.searchsorted(starts, last, side='right') - 1
    straddles = seg_first != seg_last

    # Time of each batch's center since the start of its segment, in seconds.
    # Used by the learned within-segment drift shape (`drift_shape_rank`). Each
    # sorted batch reads bfile.NT samples starting at `first`.
    center = first + np.int64(bfile.NT)//2
    ops['drift_batch_time'] = (center - starts[seg_first]) / bfile.fs

    # tmin/tmax may crop whole segments away; drop empty ones and relabel 0..k-1
    used, seg_of_batch = np.unique(seg_first, return_inverse=True)
    seg_of_batch = seg_of_batch.reshape(-1).astype('int64')
    n_segments = used.size
    ops['drift_segment_used'] = used            # original indices, for reporting
    ops['drift_segment_starts_used'] = starts[used]

    if n_segments < 2:
        raise ValueError(
            f'Chronic drift mode needs at least 2 segments within '
            f'[tmin, tmax], but only {n_segments} of '
            f'{starts.size} segment(s) contain data. Check '
            '`drift_segment_starts`, `tmin`, and `tmax`.'
            )

    counts = np.bincount(seg_of_batch, minlength=n_segments)
    logger.info(f'Chronic drift mode: {n_segments} segments, '
                f'{counts.min()}-{counts.max()} batches each '
                f'({straddles.sum()} boundary-straddling batches excluded '
                'from fingerprints).')

    return seg_of_batch, straddles, n_segments


def segment_residual_drift(ops, st, seg_of_batch, straddles, F_seg, ysamp,
                           device=torch.device('cuda'), max_batches=2000,
                           log=True):
    """Estimate per-batch residual shift relative to each segment's fingerprint.

    This tests the assumption that chronic mode makes: if drift really is
    constant within a segment, then every batch of that segment should register
    to that segment's own pooled fingerprint at ~0 offset.

    Diagnostic only, this does not modify `dshift`. Stores in `ops`:
      'drift_residual'         : (n_sampled_batches, nblocks_eff) microns
      'drift_residual_batches' : batch indices the residuals correspond to
      'drift_residual_summary' : (n_segments, nblocks_eff, 4) array of
                                 [median, std, p5, p95] in microns

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    st : np.ndarray
        Spike times array with 6 columns, as returned by `spikedetect.run`.
    seg_of_batch : np.ndarray
        Segment index of each batch, from `segment_batches`.
    straddles : np.ndarray
        Boolean mask of boundary-straddling batches, from `segment_batches`.
        Those batches mix two segments, so their residual would be a
        meaningless outlier and they are skipped.
    F_seg : np.ndarray
        Pooled per-segment fingerprints, shape (n_segments, dmax, 20).
    ysamp : np.ndarray
        Center of each vertical sampling bin.
    device : torch.device; default=torch.device('cuda').
    max_batches : int; default=2000.
        Maximum number of batches evaluated per segment. Larger segments are
        subsampled evenly, so that memory and runtime stay bounded.
    log : bool; default=True.
        If True, log the summary table and warn if the constant-per-segment
        assumption is violated. `estimate_chronic_drift` turns this off so it
        can report the residual after the learned shape model instead.

    Returns
    -------
    ops : dict

    """

    n_seg = F_seg.shape[0]
    dd = ops['binning_depth']

    # Same block boundaries as `align_block2`, so that the residuals are
    # directly comparable to `dshift`.
    ifirst, ilast = block_indices(F_seg.shape[1], ops['nblocks'])
    nblocks_eff = len(ifirst)

    # Same restricted search and x10 upsampling as the fine step of align_block2
    n = 5
    dt = np.arange(-n, n+1, 1)
    dtup = np.linspace(-n, n, 2*n*10+1)
    Kn = kernelD(dt, dtup, 1)

    sp_batch = st[:,4].astype('int64')
    all_res = []
    all_batches = []

    # One segment at a time, so that peak memory is max_batches x dmax x 20
    # floats rather than Nbatches x dmax x 20. Doing it any other way would
    # reintroduce exactly the problem chronic mode is meant to solve.
    for s in range(n_seg):
        batches = np.flatnonzero((seg_of_batch == s) & ~straddles)
        if batches.size == 0:
            continue
        if batches.size > max_batches:
            batches = batches[::int(np.ceil(batches.size/max_batches))]

        # Relabel the selected batches to a local contiguous index. `sp_batch`
        # is sorted ascending and the relabeling is monotonic, so the resulting
        # group ids are sorted as `bin_spikes` requires.
        local = np.full(seg_of_batch.size, -1, dtype='int64')
        local[batches] = np.arange(batches.size)
        lid = local[sp_batch]
        keep = lid >= 0
        Fb, _ = bin_spikes(ops, st[keep], group_id=lid[keep],
                           n_groups=batches.size)

        # mean subtract along depth, as align_block2 does
        Fb = torch.from_numpy(Fb).to(device).float()
        Fb = Fb - Fb.mean(1).unsqueeze(1)

        # The template is this segment's own fingerprint, and the segment's own
        # shift is deliberately *not* applied: the residual is measured in the
        # segment's own frame, which sidesteps any sign convention question.
        F0 = torch.from_numpy(F_seg[s]).to(device).float()
        F0 = F0 - F0.mean(0).unsqueeze(0)

        dcs = np.zeros((2*n+1, batches.size, nblocks_eff))
        for j in range(nblocks_eff):
            isub = np.arange(ifirst[j], ilast[j], 1)
            Fsub = Fb[:, isub]
            for t in range(len(dt)):
                Fs = torch.roll(Fsub, dt[t], 1)
                dcs[t,:,j] = (Fs * F0[isub]).mean(-1).mean(-1).cpu().numpy()

        res = np.zeros((batches.size, nblocks_eff))
        for j in range(nblocks_eff):
            dcup = Kn.T @ dcs[:,:,j]
            res[:,j] = dtup[np.argmax(dcup, 0)] * dd

        all_res.append(res)
        all_batches.append(batches)

    if len(all_res) == 0:
        logger.warning('No batches available for residual drift diagnostics.')
        return ops

    ops['drift_residual'] = np.concatenate(all_res, axis=0)
    ops['drift_residual_batches'] = np.concatenate(all_batches, axis=0)
    ops['drift_residual_summary'] = residual_summary(
        ops['drift_residual'], ops['drift_residual_batches'], seg_of_batch, n_seg
        )

    if log:
        _log_residual_summary(ops, ops['drift_residual_summary'], seg_of_batch,
                              straddles)

    return ops


def residual_summary(residual, batches, seg_of_batch, n_seg):
    """Per-segment [median, std, p5, p95] of residual drift, in microns.

    Parameters
    ----------
    residual : np.ndarray
        Residual shift of each sampled batch, shape (n_sampled, nblocks_eff).
    batches : np.ndarray
        Batch index of each row of `residual`.
    seg_of_batch : np.ndarray
        Segment index of each batch.
    n_seg : int
        Number of segments.

    Returns
    -------
    summary : np.ndarray
        Shape (n_seg, nblocks_eff, 4). Segments without sampled batches are NaN.

    """
    summary = np.full((n_seg, residual.shape[1], 4), np.nan)
    seg_r = seg_of_batch[batches]
    for s in range(n_seg):
        res = residual[seg_r == s]
        if res.shape[0] == 0:
            continue
        summary[s,:,0] = np.median(res, axis=0)
        summary[s,:,1] = np.std(res, axis=0)
        summary[s,:,2] = np.percentile(res, 5, axis=0)
        summary[s,:,3] = np.percentile(res, 95, axis=0)

    return summary


def _log_residual_summary(ops, summary, seg_of_batch, straddles):
    """Log the residual drift table, and warn if the assumption is violated."""

    # A residual spread below the binning resolution cannot be distinguished
    # from estimation noise, so use that (or half the interpolation scale,
    # whichever is larger) as the tolerance.
    tol = max(ops['binning_depth'], 0.5*ops['sig_interp'])
    n_seg = summary.shape[0]
    used = ops.get('drift_segment_used', np.arange(n_seg))
    has_shape = ops.get('drift_shape_weights', None) is not None

    if has_shape:
        logger.info('Within-segment residual drift after the learned shape '
                    '(worst block per segment):')
    else:
        logger.info('Within-segment residual drift (worst block per segment):')
    logger.info('  seg  batches  median(um)       p5-p95(um)  max|res|(um)')
    spread_bad = []
    bias_bad = []
    for s in range(n_seg):
        if np.all(np.isnan(summary[s])):
            continue
        nb = int(np.sum((seg_of_batch == s) & ~straddles))
        spread = summary[s,:,3] - summary[s,:,2]
        j = int(np.argmax(spread))
        med, p5, p95 = summary[s,j,0], summary[s,j,2], summary[s,j,3]
        mx = np.max(np.abs(summary[s][:,[0,2,3]]))
        logger.info(f'  {used[s]:>3}  {nb:>7}  {med:>+10.2f}  '
                    f'{p5:>+6.2f}..{p95:>+6.2f}  {mx:>12.2f}')
        if spread[j] > tol:
            spread_bad.append((int(used[s]), spread[j]))
        if np.max(np.abs(summary[s,:,0])) > tol:
            bias_bad.append((int(used[s]), np.max(np.abs(summary[s,:,0]))))

    if len(spread_bad) > 0:
        seg_txt = ', '.join([f'{s} ({v:.1f} um)' for s, v in spread_bad])
        if has_shape:
            model_txt = (
                'The learned within-segment shape does not capture the drift '
                'in those segments. Try increasing `drift_shape_rank` or '
                '`drift_shape_nbasis`, splitting them into shorter segments, '
                )
        else:
            model_txt = (
                'Drift is not constant within those segments, so the chronic '
                'model is a poor fit for them. Try a learned within-segment '
                'shape (`drift_shape_rank = 1`), splitting them into shorter '
                'segments, '
                )
        warnings.warn(
            f'Within-segment residual drift exceeds {tol:.1f} um (p5-p95) for '
            f'segment(s): {seg_txt}. {model_txt}or use standard per-batch '
            'drift correction (`drift_segment_starts = None`).',
            UserWarning
            )

    if len(bias_bad) > 0:
        seg_txt = ', '.join([f'{s} ({v:+.1f} um)' for s, v in bias_bad])
        warnings.warn(
            f'Median within-segment residual exceeds {tol:.1f} um for '
            f'segment(s): {seg_txt}. A consistently non-zero median, as '
            'opposed to a large spread, means the pooled registration is '
            'biased for those segments. That usually indicates that the '
            'fingerprint content changed rather than just its position '
            '(units lost or gained, or a gain change). Inspect '
            'drift_segments.png before trusting across-segment unit identity.',
            UserWarning
            )


def gaussian_time_basis(t, time_range, n_basis):
    """Gaussian bumps evenly spaced across `time_range`.

    Adjacent bumps are one standard deviation apart, so they overlap enough
    that weighting them gives a smooth curve, and no weighting can produce
    variation faster than the bump width.

    Parameters
    ----------
    t : np.ndarray
        Times at which to evaluate the basis, in seconds.
    time_range : tuple of float
        (first, last) bump center.
    n_basis : int
        Number of bumps, at least 2.

    Returns
    -------
    phi : np.ndarray
        Shape (len(t), n_basis).

    """
    t = np.asarray(t, dtype='float64')
    t0, t1 = float(time_range[0]), float(time_range[1])
    centers = np.linspace(t0, t1, n_basis)
    width = (t1 - t0) / (n_basis - 1)
    if width <= 0:
        width = 1.0
    return np.exp(-(t[:,None] - centers)**2 / (2*width**2))


def _weighted_ridge(X, w, Y, ridge):
    """Weighted ridge regressions of each column of `Y` on a shared design.

    X is (n, P), w and Y are (n, J). Returns coefficients of shape (J, P). The
    first column of X is an intercept and is not penalized. The penalty is
    scaled by the average diagonal of X'WX, so `ridge` does not depend on
    units or on the number of rows.

    """
    P = X.shape[1]
    XtWX = np.einsum('np,nj,nq->jpq', X, w, X)
    XtWy = np.einsum('np,nj->jp', X, w*Y)
    scale = np.trace(XtWX, axis1=1, axis2=2) / P
    penalty = np.eye(P)
    penalty[0,0] = 0
    lhs = XtWX + ridge*scale[:,None,None]*penalty + 1e-12*np.eye(P)
    return np.linalg.solve(lhs, XtWy[...,None])[...,0]


def _huber_weights(err, resolution, c=1.345):
    """Huber IRLS weights, with the scale taken from the median absolute deviation.

    Per-batch residuals are argmax estimates, so a small fraction are far off
    (for example batches with few spikes). `resolution` floors the scale,
    because quantized residuals can have a median absolute deviation of 0.

    """
    scale = 1.4826 * np.median(np.abs(err - np.median(err)))
    scale = max(scale, resolution)
    a = np.abs(err) / scale
    return np.where(a <= c, 1.0, c / np.maximum(a, 1e-12))


def _normalize_shapes(W, phi_grid, fallback=None):
    """Make shapes orthogonal with unit RMS over the time range, and fix signs.

    The model is invariant to rescaling a shape and its amplitudes in
    opposite directions, so this pins down a unique, interpretable scale:
    amplitudes are then in microns per unit-RMS shape.

    """
    G = phi_grid @ W.T
    n = G.shape[0]
    _, R = np.linalg.qr(G / np.sqrt(n))
    if np.any(np.abs(np.diag(R)) < 1e-12):
        # Degenerate (e.g. no within-segment drift at all), keep previous shapes.
        return W if fallback is None else fallback
    W = np.linalg.solve(R.T, W)

    # Sign convention: shapes increase from the start to the end of the range,
    # or are positive at their peak if they start and end at the same level.
    G = phi_grid @ W.T
    for k in range(W.shape[0]):
        sign = np.sign(G[-1,k] - G[0,k])
        if sign == 0:
            sign = np.sign(G[np.argmax(np.abs(G[:,k])), k])
        if sign < 0:
            W[k] = -W[k]

    return W


def fit_drift_shape(residual, seg, t, n_seg, time_range, rank=1, n_basis=8,
                    resolution=0.5, max_iter=50, tol=1e-6, ridge=1e-3):
    """Fit a low-rank, smooth within-segment drift model to residual drift.

    The model for row `n` (a batch in segment `s`) and depth block `j` is

        residual[n, j] = intercept[s, j]
                         + sum_k amplitude[s, j, k] * g_{s,k}(t[n])

    where each segment has its own shapes `g_{s,k}`, shared by all depth
    blocks of that segment. So every segment can follow its own course, while
    the blocks of a segment move together, each by its own amount. The shapes
    are weighted sums of Gaussian bumps across time, `g_{s,k} = sum_m
    weights[s, k, m] * phi_m`. The basis is mean-subtracted over the time
    range, so that a constant cannot hide in a shape; per-segment levels go in
    `intercept`.

    Segments are fit independently. Within a segment, fitting alternates
    between weighted least squares for (intercept, amplitude) given the
    shapes, and for the shape weights given the amplitudes, with Huber weights
    updated in between. It is initialized with the leading singular vectors of
    unconstrained per-block bump fits.

    Parameters
    ----------
    residual : np.ndarray
        Residual shift in microns, shape (n, J).
    seg : np.ndarray
        Segment index of each row, in [0, n_seg).
    t : np.ndarray
        Time of each row within its segment. `apply_drift_shape` uses the
        position within the segment (0 to 1), so that the bumps span each
        segment separately.
    n_seg : int
        Number of segments. Segments without rows get zero intercept,
        amplitude and shape weights.
    time_range : tuple of float
        Time range covered by the basis. Should include every time the model
        will be evaluated at.
    rank : int; default=1.
        Number of shapes per segment. `rank = J` lets every block of a
        segment follow its own course.
    n_basis : int; default=8.
        Number of Gaussian bumps per shape.
    resolution : float; default=0.5.
        Smallest meaningful residual, in microns. Floors the robust scale.
    max_iter : int; default=50.
    tol : float; default=1e-6.
        Stop when shape weights change by less than this.
    ridge : float; default=1e-3.
        Relative ridge penalty on amplitudes and shape weights.

    Returns
    -------
    fit : dict
        'intercept' (n_seg, J), 'amplitude' (n_seg, J, rank) in microns per
        unit-RMS shape, 'weights' (n_seg, rank, n_basis), 'basis_mean'
        (n_basis,), 'time_range' (2,), 'n_iter' (n_seg,).

    """
    residual = np.asarray(residual, dtype='float64')
    if residual.ndim == 1:
        residual = residual[:,None]
    seg = np.asarray(seg, dtype='int64')
    N, J = residual.shape
    K, M = int(rank), int(n_basis)
    if M < 2:
        raise ValueError(f'`n_basis` must be at least 2, got {M}.')
    if K < 1 or K > min(M, J):
        raise ValueError(
            f'`rank` must be between 1 and min(n_basis, n_blocks) '
            f'= {min(M, J)}, got {K}.'
            )

    time_range = (float(time_range[0]), float(time_range[1]))
    t_grid = np.linspace(time_range[0], time_range[1], 512)
    phi_grid = gaussian_time_basis(t_grid, time_range, M)
    basis_mean = phi_grid.mean(0)
    phi_grid = phi_grid - basis_mean
    phi = gaussian_time_basis(t, time_range, M) - basis_mean

    intercept = np.zeros((n_seg, J))
    amplitude = np.zeros((n_seg, J, K))
    W = np.zeros((n_seg, K, M))
    n_iter = np.zeros(n_seg, dtype='int64')
    for s in range(n_seg):
        idx = np.flatnonzero(seg == s)
        if idx.size == 0:
            continue
        intercept[s], amplitude[s], W[s], n_iter[s] = _fit_segment_shape(
            residual[idx], phi[idx], phi_grid, K, resolution, max_iter, tol,
            ridge
            )

    return {
        'intercept': intercept, 'amplitude': amplitude, 'weights': W,
        'basis_mean': basis_mean, 'time_range': np.array(time_range),
        'n_iter': n_iter,
        }


def _fit_segment_shape(Y, P, phi_grid, K, resolution, max_iter, tol, ridge):
    """Fit `K` shapes shared by the blocks of one segment, see `fit_drift_shape`.

    Y is the residual (n, J), P the mean-subtracted basis at each row (n, M).
    Returns intercept (J,), amplitude (J, K), weights (K, M) and the number of
    iterations.

    """
    n, J = Y.shape
    M = P.shape[1]
    ones = np.ones((n, 1))
    w = np.ones((n, J))

    # Initialize with the dominant shapes of unconstrained per-block fits.
    B = _weighted_ridge(np.concatenate([ones, P], 1), w, Y, ridge)[:,1:]
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    W = _normalize_shapes(Vt[:K], phi_grid)

    def block_step(W, w):
        G = P @ W.T
        coef = _weighted_ridge(np.concatenate([ones, G], 1), w, Y, ridge)
        return G, coef[:,0], coef[:,1:]

    n_iter = 0
    for n_iter in range(1, max_iter+1):
        # intercepts and amplitudes, given the shapes
        G, intercept, amplitude = block_step(W, w)
        err = Y - intercept - G @ amplitude.T
        w = _huber_weights(err, resolution)

        # Shape weights, given intercepts and amplitudes. The design row for
        # (n, j) is kron(amplitude[j], P[n]).
        ys = Y - intercept
        PtWP = np.einsum('nm,nj,np->jmp', P, w, P)
        PtWy = np.einsum('nm,nj->jm', P, w*ys)
        XtX = np.einsum('jk,jl,jmp->kmlp', amplitude, amplitude,
                        PtWP).reshape(K*M, K*M)
        Xty = np.einsum('jk,jm->km', amplitude, PtWy).reshape(K*M)
        lam = ridge*np.trace(XtX)/(K*M) + 1e-12
        W_new = np.linalg.solve(XtX + lam*np.eye(K*M), Xty).reshape(K, M)
        W_new = _normalize_shapes(W_new, phi_grid, fallback=W)

        change = np.max(np.abs(W_new - W))
        W = W_new
        if change < tol:
            break

    G, intercept, amplitude = block_step(W, w)

    # Order shapes by how much drift they account for.
    order = np.argsort(-np.sum(amplitude**2, axis=0))

    return intercept, amplitude[:,order], W[order], n_iter


def drift_shape_curves(fit, t):
    """Evaluate each segment's shapes at times `t`.

    Returns (n_seg, len(t), rank).

    """
    W = fit['weights']
    phi = gaussian_time_basis(t, fit['time_range'], W.shape[-1]) - fit['basis_mean']
    return np.einsum('nm,skm->snk', phi, W)


def drift_shape_model(fit, seg, t):
    """Within-segment drift predicted by `fit`, in microns. Returns (len(t), J)."""
    seg = np.asarray(seg, dtype='int64')
    W = fit['weights']
    phi = gaussian_time_basis(t, fit['time_range'], W.shape[-1]) - fit['basis_mean']
    G = np.einsum('nm,nkm->nk', phi, W[seg])
    return fit['intercept'][seg] + np.einsum('njk,nk->nj', fit['amplitude'][seg], G)


def segment_phase(t, seg_of_batch, n_seg):
    """Position of each batch within its segment, from 0 (first) to 1 (last).

    Parameters
    ----------
    t : np.ndarray
        Time of each batch, in any units that increase within a segment.
    seg_of_batch : np.ndarray
        Segment index of each batch.
    n_seg : int

    Returns
    -------
    phase : np.ndarray
        Same shape as `t`. Segments with a single batch get 0.5.

    """
    t = np.asarray(t, dtype='float64')
    phase = np.full(t.shape, 0.5)
    for s in range(n_seg):
        idx = seg_of_batch == s
        if not np.any(idx):
            continue
        t0, t1 = t[idx].min(), t[idx].max()
        if t1 > t0:
            phase[idx] = (t[idx] - t0) / (t1 - t0)

    return phase


def apply_drift_shape(ops, seg_of_batch, n_seg):
    """Fit the learned within-segment shapes and evaluate them for every batch.

    Uses the per-batch residuals from `segment_residual_drift`. Those are
    measured against each segment's own pooled fingerprint in the same roll
    convention as `align_block2`, so the total shift of a batch is the segment
    shift plus the modeled residual.

    Stores the fit in `ops` (keys starting with 'drift_shape_'), keeps the raw
    residual in 'drift_residual_raw', and replaces 'drift_residual' and
    'drift_residual_summary' with the residual left after the model.

    Returns
    -------
    ops : dict
    model : np.ndarray
        Within-segment drift for every batch, (Nbatches, nblocks_eff) microns.

    """
    rank = int(ops['drift_shape_rank'])
    n_basis = int(ops.get('drift_shape_nbasis', 8))
    residual = ops['drift_residual']
    batches = ops['drift_residual_batches']
    # The bumps span each segment separately: time is the position within the
    # segment, from its first batch (0) to its last (1), computed over every
    # batch so that the model is never extrapolated.
    t_all = segment_phase(ops['drift_batch_time'], seg_of_batch, n_seg)
    ops['drift_batch_phase'] = t_all
    time_range = (0.0, 1.0)
    resolution = 0.1 * ops['binning_depth']   # the x10 upsampled search grid
    seg_r = seg_of_batch[batches]
    t_r = t_all[batches]
    J = residual.shape[1]

    # Held-out check: fit on every other sampled batch, and compare the error
    # on the rest against a constant per segment fit to the same batches.
    improvement = np.nan
    if batches.size >= 4:
        train = (np.arange(batches.size) % 2) == 0
        test = ~train
        fit_train = fit_drift_shape(
            residual[train], seg_r[train], t_r[train], n_seg, time_range,
            rank=rank, n_basis=n_basis, resolution=resolution
            )
        const = np.zeros((n_seg, J))
        for s in range(n_seg):
            r = residual[train][seg_r[train] == s]
            if r.shape[0] > 0:
                const[s] = np.median(r, axis=0)
        pred = drift_shape_model(fit_train, seg_r[test], t_r[test])
        err_model = np.abs(residual[test] - pred).mean()
        err_const = np.abs(residual[test] - const[seg_r[test]]).mean()
        if err_const > 0:
            improvement = 1 - err_model/err_const

    fit = fit_drift_shape(residual, seg_r, t_r, n_seg, time_range, rank=rank,
                          n_basis=n_basis, resolution=resolution)
    model = drift_shape_model(fit, seg_of_batch, t_all)

    t_grid = np.linspace(time_range[0], time_range[1], 200)
    ops['drift_shape_intercept'] = fit['intercept']
    ops['drift_shape_amplitude'] = fit['amplitude']
    ops['drift_shape_weights'] = fit['weights']
    ops['drift_shape_basis_mean'] = fit['basis_mean']
    ops['drift_shape_time_range'] = fit['time_range']
    ops['drift_shape_time_grid'] = t_grid
    ops['drift_shape_curves'] = drift_shape_curves(fit, t_grid)
    ops['drift_shape_model'] = model
    ops['drift_shape_heldout_improvement'] = improvement

    ops['drift_residual_raw'] = residual
    ops['drift_residual'] = residual - model[batches]
    ops['drift_residual_summary'] = residual_summary(
        ops['drift_residual'], batches, seg_of_batch, n_seg
        )

    used = ops.get('drift_segment_used', np.arange(n_seg))
    ranges = []
    for s in range(n_seg):
        m = model[seg_of_batch == s]
        ranges.append(f'{used[s]}: {np.max(m.max(0) - m.min(0)):.1f}')
    logger.info(f'Learned within-segment drift shapes: rank {rank} per '
                f'segment, {n_basis} Gaussian bumps spanning each segment '
                f'(at most {fit["n_iter"].max()} iterations).')
    logger.info('Within-segment drift range per segment (um, worst block): '
                + ', '.join(ranges))
    if np.isfinite(improvement):
        logger.info(f'Held-out batches: mean |residual| {100*improvement:+.0f}% '
                    'lower than with a constant per segment.')
        if improvement <= 0:
            warnings.warn(
                'The learned within-segment drift shape does not reduce the '
                'residual on held-out batches, so it is most likely fitting '
                'noise. Consider `drift_shape_rank = 0`.',
                UserWarning
                )

    return ops, model


def estimate_chronic_drift(ops, st, bfile, device=torch.device('cuda')):
    """Estimate drift per recording segment, optionally with a learned shape.

    Parameters
    ----------
    ops : dict
        Must contain `drift_segment_starts`. `drift_shape_rank > 0` adds the
        learned within-segment shape, `drift_segment_diagnostics` controls
        whether the residual table is logged.
    st : np.ndarray
        Spike times array with 6 columns, as returned by `spikedetect.run`.
    bfile : kilosort.io.BinaryFiltered
    device : torch.device; default=torch.device('cuda').

    Returns
    -------
    imin : np.ndarray
        Shift of each batch and block in units of depth bins, shape
        (Nbatches, nblocks_eff).
    yblk : np.ndarray
        Block center depths.
    ops : dict

    """
    seg_of_batch, straddles, n_seg = segment_batches(ops, bfile)
    sp_batch = st[:,4].astype('int64')
    keep = ~straddles[sp_batch]
    gid = seg_of_batch[sp_batch[keep]]        # already sorted ascending

    # all spikes of a segment are pooled into a single fingerprint
    F, ysamp = bin_spikes(ops, st[keep], group_id=gid, n_groups=n_seg)

    # Never smooth the dot products across segment boundaries: a
    # segment-to-segment step is real signal, not noise. The axis order is
    # (correlation, time, block).
    smoothing = list(ops['drift_smoothing'])
    smoothing[1] = 0.0

    imin_seg, yblk, _, _ = align_block2(
        F, ysamp, ops, device=device, drift_smoothing=smoothing
        )

    # Broadcast the per-segment shift back to every batch. `dshift` keeps its
    # original (Nbatches, nblocks_eff) shape, so nothing downstream needs to
    # know that chronic mode was used.
    imin = imin_seg[seg_of_batch]
    ops['batch_to_segment'] = seg_of_batch
    ops['drift_segment_shift'] = imin_seg * ops['binning_depth']
    ops['drift_segment_fingerprints'] = F

    rank = int(ops.get('drift_shape_rank', 0) or 0)
    diagnostics = ops.get('drift_segment_diagnostics', True)
    if diagnostics or rank > 0:
        ops = segment_residual_drift(ops, st, seg_of_batch, straddles, F, ysamp,
                                     device=device, log=False)

    if rank > 0:
        if 'drift_residual' in ops:
            ops, model = apply_drift_shape(ops, seg_of_batch, n_seg)
            imin = imin + model / ops['binning_depth']
        else:
            warnings.warn('No batches were available to fit the learned '
                          'within-segment drift shape, so it was skipped.',
                          UserWarning)

    if diagnostics and 'drift_residual_summary' in ops:
        _log_residual_summary(ops, ops['drift_residual_summary'], seg_of_batch,
                              straddles)

    return imin, yblk, ops


def drift_row_ids(dshift, n_chan, max_bytes=256e6):
    """Label identical rows of `dshift`, for caching drift matrices.

    Returns None if caching one `Nchan x Nchan` float32 matrix per distinct row
    would exceed `max_bytes`, which is the case whenever drift varies within
    segments.

    """
    rows, ids = np.unique(dshift, axis=0, return_inverse=True)
    max_entries = max(1, int(max_bytes // (4 * n_chan**2)))
    if rows.shape[0] > max_entries:
        return None
    return ids.reshape(-1).astype('int64')


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
        ops['batch_to_segment'] = None
        ops['drift_row_id'] = None
        logger.info('nblocks = 0, skipping drift correction')
        return ops, None

    # the first step is to extract all spikes using the universal templates.
    # PC features are not needed for drift estimation, so don't store them.
    st, _, ops  = spikedetect.run(
        ops, bfile, device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose, save_features=False
        )

    segment_starts = ops.get('drift_segment_starts', None)
    if segment_starts is None:
        # --- standard per-batch estimation ---
        # spikes are binned by amplitude and y-position to construct a
        # "fingerprint" for each batch
        F, ysamp = bin_spikes(ops, st)

        # the fingerprints are iteratively aligned to each other vertically
        imin, yblk, _, _ = align_block2(F, ysamp, ops, device=device)
        ops['batch_to_segment'] = None
    else:
        # --- chronic mode: one shift per (segment, block), optionally plus a
        # learned within-segment shape ---
        imin, yblk, ops = estimate_chronic_drift(ops, st, bfile, device=device)

    # imin contains the shifts for each batch, in units of discrete bins
    # multiply back with binning_depth for microns
    dshift = imin * ops['binning_depth']

    # Batches with identical shifts can share one drift matrix (see
    # `io.BinaryFiltered._drift_whiten`). Per-batch drift is never cached, so
    # memory use is unchanged when chronic mode is off.
    if segment_starts is None:
        ops['drift_row_id'] = None
    else:
        ops['drift_row_id'] = drift_row_ids(dshift, len(ops['xc']))

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
