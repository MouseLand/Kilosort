"""Tests for chronic drift mode, where drift is constant within each segment.

These are self-contained: they build small synthetic spike sets rather than
relying on the downloaded test data, so they run without `--download`.

"""

import numpy as np
import pytest
import torch
from scipy.sparse import coo_matrix

from kilosort import datashift, io
from kilosort.parameters import DEFAULT_SETTINGS


def make_ops(nblocks=1, binning_depth=5, n_chans=20, chan_spacing=20):
    """Minimal `ops` with the keys the drift estimation functions read."""
    yc = np.arange(n_chans, dtype='float64') * chan_spacing
    return {
        'yc': yc,
        'xc': np.zeros_like(yc),
        'binning_depth': binning_depth,
        'Th_universal': DEFAULT_SETTINGS['Th_universal'],
        'nblocks': nblocks,
        'drift_smoothing': DEFAULT_SETTINGS['drift_smoothing'],
        'sig_interp': DEFAULT_SETTINGS['sig_interp'],
        'Nbatches': 0,
        }


def bin_spikes_reference(ops, st):
    """The original per-batch implementation, kept as a regression reference."""
    ymin = ops['yc'].min()
    ymax = ops['yc'].max()
    dd = ops['binning_depth']
    dmin = ymin - 1
    dmax = 1 + np.ceil((ymax-dmin)/dd).astype('int32')
    Nbatches = ops['Nbatches']
    batch_id = st[:,4].copy()

    F = np.zeros((Nbatches, dmax, 20))
    for t in range(Nbatches):
        ix = (batch_id==t).nonzero()[0]
        sst = st[ix]
        dep = sst[:,1] - dmin
        amp = np.log10(np.minimum(99, sst[:,2])) - np.log10(ops['Th_universal'])
        amp = amp / (np.log10(100)-np.log10(ops['Th_universal']))
        rows = (dep/dd).astype('int32')
        cols = (1e-5 + amp * 20).astype('int32')
        cou = np.ones(len(ix))
        M = coo_matrix((cou, (rows, cols)), (dmax, 20))
        F[t] = np.log2(1+M.todense())

    ysamp = dmin + dd * np.arange(dmax) - dd/2
    return F, ysamp


def make_spikes(ops, n_batches, shift_of_batch, n_neurons=12, per_batch=400,
                seed=0):
    """Synthetic spikes from fixed neurons, displaced by a per-batch shift.

    Returns `st` with the columns the drift code uses: depth in column 1,
    amplitude in column 2, and batch index (ascending) in column 4.

    """
    rng = np.random.default_rng(seed)
    ymin, ymax = ops['yc'].min(), ops['yc'].max()
    # keep neurons away from the edges so shifting does not push them off
    depths = np.linspace(ymin + 60, ymax - 60, n_neurons)
    amps = rng.uniform(12, 80, n_neurons)
    rates = rng.uniform(0.5, 1.5, n_neurons)
    rates = rates/rates.sum()

    st = []
    for b in range(n_batches):
        which = rng.choice(n_neurons, size=per_batch, p=rates)
        col = np.zeros((per_batch, 6))
        col[:,1] = depths[which] + shift_of_batch[b] + rng.normal(0, 1.5, per_batch)
        col[:,2] = amps[which] * rng.uniform(0.9, 1.1, per_batch)
        col[:,4] = b
        st.append(col)

    return np.concatenate(st, axis=0)


class StubFile:
    """Stands in for `io.BinaryFiltered` in `segment_batches`."""
    def __init__(self, NT=60000, imin=0, n_batches=10, batch_downsampling=1):
        self.NT = NT
        self.imin = imin
        self.n_batches = n_batches
        self.batch_downsampling = batch_downsampling


class TestBinSpikes:

    def test_matches_reference(self):
        # The refactored grouping must not change the default per-batch result.
        ops = make_ops()
        ops['Nbatches'] = 8
        shifts = np.zeros(8)
        st = make_spikes(ops, 8, shifts, seed=1)

        F_ref, ysamp_ref = bin_spikes_reference(ops, st)
        F_new, ysamp_new = datashift.bin_spikes(ops, st)

        assert np.array_equal(F_ref, F_new)
        assert np.array_equal(ysamp_ref, ysamp_new)

    def test_empty_group(self):
        # A group with no spikes should give an all-zero fingerprint, not fail.
        ops = make_ops()
        ops['Nbatches'] = 4
        st = make_spikes(ops, 4, np.zeros(4), seed=2)
        st = st[st[:,4] != 2]   # drop every spike from batch 2

        F, _ = datashift.bin_spikes(ops, st)
        assert F.shape[0] == 4
        assert np.all(F[2] == 0)
        assert F[1].sum() > 0

    def test_pooling_sums_counts(self):
        # Pooling two batches must equal binning both batches' spikes together.
        ops = make_ops()
        ops['Nbatches'] = 4
        st = make_spikes(ops, 4, np.zeros(4), seed=3)

        F_batch, _ = datashift.bin_spikes(ops, st)
        group_id = (st[:,4]//2).astype('int64')      # {0,1} -> 0, {2,3} -> 1
        F_pool, _ = datashift.bin_spikes(ops, st, group_id=group_id, n_groups=2)

        assert F_pool.shape[0] == 2
        for s in range(2):
            counts = (2**F_batch[2*s] - 1) + (2**F_batch[2*s+1] - 1)
            assert np.allclose(F_pool[s], np.log2(1 + counts))

    def test_unsorted_group_id_raises(self):
        ops = make_ops()
        ops['Nbatches'] = 4
        st = make_spikes(ops, 4, np.zeros(4), seed=4)
        group_id = np.zeros(st.shape[0], dtype='int64')
        group_id[0] = 1     # out of order

        with pytest.raises(ValueError, match='non-decreasing'):
            datashift.bin_spikes(ops, st, group_id=group_id, n_groups=2)


class TestLoadDriftSegments:

    def test_newline_separated(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0\n100\n250\n')
        assert np.array_equal(io.load_drift_segments(p), [0, 100, 250])

    def test_comma_and_space_separated(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0, 100 , 250')
        assert np.array_equal(io.load_drift_segments(p), [0, 100, 250])
        p.write_text('0 100 250')
        assert np.array_equal(io.load_drift_segments(p), [0, 100, 250])

    def test_list_input(self):
        assert np.array_equal(io.load_drift_segments([0, 5, 9]), [0, 5, 9])

    def test_prepends_zero(self, tmp_path):
        # A file listing only the starts of days 2..N is still usable.
        p = tmp_path / 'segments.txt'
        p.write_text('100\n250\n')
        assert np.array_equal(io.load_drift_segments(p), [0, 100, 250])

    def test_rejects_non_monotonic(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0 250 100')
        with pytest.raises(ValueError, match='increasing'):
            io.load_drift_segments(p)

    def test_rejects_duplicates(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0 100 100')
        with pytest.raises(ValueError, match='increasing'):
            io.load_drift_segments(p)

    def test_rejects_non_integer(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0 100.5')
        with pytest.raises(ValueError, match='integer'):
            io.load_drift_segments(p)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match='non-negative'):
            io.load_drift_segments([-10, 0, 100])

    def test_rejects_garbage(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('day1 day2')
        with pytest.raises(ValueError, match='parse'):
            io.load_drift_segments(p)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileExistsError):
            io.load_drift_segments(tmp_path / 'nope.txt')


class TestSegmentBatches:

    def test_boundary_on_batch_edge(self):
        # Boundary at exactly 4 batches: no batch should straddle.
        ops = make_ops()
        ops['drift_segment_starts'] = np.array([0, 4*60000])
        bfile = StubFile(NT=60000, imin=0, n_batches=10)

        seg, straddles, n_seg = datashift.segment_batches(ops, bfile)
        assert n_seg == 2
        assert np.array_equal(seg, [0,0,0,0,1,1,1,1,1,1])
        assert not straddles.any()

    def test_boundary_mid_batch(self):
        # Boundary halfway through batch 4: that batch straddles and is
        # assigned to the earlier segment, since that is where it starts.
        ops = make_ops()
        ops['drift_segment_starts'] = np.array([0, 4*60000 + 30000])
        bfile = StubFile(NT=60000, imin=0, n_batches=10)

        seg, straddles, n_seg = datashift.segment_batches(ops, bfile)
        assert n_seg == 2
        assert np.array_equal(seg, [0,0,0,0,0,1,1,1,1,1])
        assert straddles[4]
        assert straddles.sum() == 1

    def test_tmin_offset(self):
        # imin shifts every batch's sample range, so the boundary moves with it.
        ops = make_ops()
        ops['drift_segment_starts'] = np.array([0, 5*60000])
        bfile = StubFile(NT=60000, imin=2*60000, n_batches=10)

        seg, straddles, n_seg = datashift.segment_batches(ops, bfile)
        # batch b starts at sample (2+b)*NT, so the boundary falls at b == 3
        assert np.array_equal(seg, [0,0,0,1,1,1,1,1,1,1])
        assert not straddles.any()

    def test_batch_downsampling(self):
        # Each sorted batch advances by NT*batch_downsampling raw samples.
        ops = make_ops()
        ops['drift_segment_starts'] = np.array([0, 4*60000])
        bfile = StubFile(NT=60000, imin=0, n_batches=6, batch_downsampling=2)

        seg, straddles, n_seg = datashift.segment_batches(ops, bfile)
        # batch b covers [b*120000, (b+1)*120000), so batch 1 straddles 240000
        assert np.array_equal(seg, [0,0,1,1,1,1])
        assert not straddles.any()

    def test_cropped_segment_is_dropped_and_relabeled(self):
        # tmin crops away segment 0 entirely; the rest relabel from 0.
        ops = make_ops()
        ops['drift_segment_starts'] = np.array([0, 2*60000, 6*60000])
        bfile = StubFile(NT=60000, imin=2*60000, n_batches=8)

        seg, straddles, n_seg = datashift.segment_batches(ops, bfile)
        assert n_seg == 2
        assert np.array_equal(np.unique(seg), [0, 1])
        # original indices are kept for reporting
        assert np.array_equal(ops['drift_segment_used'], [1, 2])
        assert np.array_equal(ops['drift_segment_starts_used'],
                              [2*60000, 6*60000])

    def test_single_surviving_segment_raises(self):
        ops = make_ops()
        ops['drift_segment_starts'] = np.array([0, 100*60000])
        bfile = StubFile(NT=60000, imin=0, n_batches=10)

        with pytest.raises(ValueError, match='at least 2 segments'):
            datashift.segment_batches(ops, bfile)


class TestAlignBlock2:

    def test_shape_follows_F_not_ops(self):
        # align_block2 must size its output from F, so that it can be reused
        # on per-segment fingerprints.
        ops = make_ops(nblocks=2)
        ops['Nbatches'] = 999     # deliberately wrong, must be ignored
        n_units = 6
        st = make_spikes(ops, n_units, np.zeros(n_units), seed=5)
        F, ysamp = datashift.bin_spikes(
            ops, st, group_id=st[:,4].astype('int64'), n_groups=n_units
            )

        imin, yblk, _, _ = datashift.align_block2(
            F, ysamp, ops, device=torch.device('cpu')
            )
        assert imin.shape == (n_units, 2*ops['nblocks'] - 1)
        assert yblk.shape == (2*ops['nblocks'] - 1,)

    def test_too_few_fingerprints_raises(self):
        ops = make_ops()
        st = make_spikes(ops, 1, np.zeros(1), seed=6)
        F, ysamp = datashift.bin_spikes(
            ops, st, group_id=st[:,4].astype('int64'), n_groups=1
            )
        with pytest.raises(ValueError, match='at least 2 fingerprints'):
            datashift.align_block2(F, ysamp, ops, device=torch.device('cpu'))


class TestChronicEstimation:
    """The estimation itself, on spikes with a known step between segments."""

    binning_depth = 5
    true_step = 20.0            # microns, segment 1 relative to segment 0
    n_batches = 40
    boundary = 20               # first batch of segment 1

    def build(self, seed=7, per_batch=800):
        ops = make_ops(nblocks=1, binning_depth=self.binning_depth)
        ops['Nbatches'] = self.n_batches
        shifts = np.zeros(self.n_batches)
        shifts[self.boundary:] = self.true_step
        st = make_spikes(ops, self.n_batches, shifts, per_batch=per_batch,
                         seed=seed)
        seg_of_batch = (np.arange(self.n_batches) >= self.boundary).astype('int64')
        return ops, st, seg_of_batch

    def test_recovers_step_between_segments(self):
        ops, st, seg_of_batch = self.build()
        group_id = seg_of_batch[st[:,4].astype('int64')]
        F, ysamp = datashift.bin_spikes(ops, st, group_id=group_id, n_groups=2)

        smoothing = list(ops['drift_smoothing'])
        smoothing[1] = 0.0
        imin_seg, _, _, _ = datashift.align_block2(
            F, ysamp, ops, device=torch.device('cpu'), drift_smoothing=smoothing
            )

        # The estimated separation between the two segments should match the
        # true step, up to one depth bin.
        step_bins = np.abs(imin_seg[1,0] - imin_seg[0,0])
        assert np.isclose(step_bins * self.binning_depth, self.true_step,
                          atol=self.binning_depth)

        # Applying the estimated shifts should bring the fingerprints into
        # register, whatever the sign convention.
        def corr(a, b):
            a = a - a.mean(); b = b - b.mean()
            return float((a*b).sum()/np.sqrt((a**2).sum()*(b**2).sum()))

        before = corr(F[0], F[1])
        aligned = [np.roll(F[s], int(np.round(imin_seg[s,0])), axis=0)
                   for s in range(2)]
        after = corr(aligned[0], aligned[1])
        assert after > before
        assert after > 0.9

    def test_pooling_beats_per_batch_estimation(self):
        # The point of the mode: one estimate per segment removes the jitter
        # of the per-batch estimates while still recovering the true step.
        # Use few spikes per batch, which is the realistic regime: on a real
        # probe a 2 s batch has well under one spike per histogram bin.
        ops, st, seg_of_batch = self.build(per_batch=40, seed=9)

        F_batch, ysamp = datashift.bin_spikes(ops, st)
        imin_batch, _, _, _ = datashift.align_block2(
            F_batch, ysamp, ops, device=torch.device('cpu')
            )

        # Within a segment the true shift is constant, so any spread in the
        # per-batch estimate is estimation noise.
        noise = np.concatenate([
            imin_batch[seg_of_batch == s, 0]
            - np.median(imin_batch[seg_of_batch == s, 0])
            for s in (0, 1)
            ])
        assert noise.std() > 0    # there is per-batch jitter to remove

        group_id = seg_of_batch[st[:,4].astype('int64')]
        F_seg, _ = datashift.bin_spikes(ops, st, group_id=group_id, n_groups=2)
        smoothing = list(ops['drift_smoothing'])
        smoothing[1] = 0.0
        imin_seg, _, _, _ = datashift.align_block2(
            F_seg, ysamp, ops, device=torch.device('cpu'),
            drift_smoothing=smoothing
            )

        # Broadcasting the segment estimate gives an exactly flat trace within
        # each segment: the jitter above is gone by construction.
        dshift = imin_seg[seg_of_batch]
        for s in (0, 1):
            rows = dshift[seg_of_batch == s]
            assert np.all(rows == rows[0])

        # ...and the step is still recovered, from the same data that made the
        # per-batch estimates jitter.
        seg_step = np.abs(imin_seg[1,0] - imin_seg[0,0])
        assert np.isclose(seg_step * self.binning_depth, self.true_step,
                          atol=self.binning_depth)

    def test_residual_diagnostic_is_small_when_assumption_holds(self):
        ops, st, seg_of_batch = self.build()
        group_id = seg_of_batch[st[:,4].astype('int64')]
        F_seg, _ = datashift.bin_spikes(ops, st, group_id=group_id, n_groups=2)

        ops = datashift.segment_residual_drift(
            ops, st, seg_of_batch, F_seg, device=torch.device('cpu')
            )

        residual = ops['drift_residual']
        assert residual.shape == (self.n_batches, 2*ops['nblocks'] - 1)
        assert ops['drift_residual_summary'].shape == (2, 1, 4)
        # Drift really is constant within each segment here, so the residual
        # should be centered near zero.
        assert np.abs(np.median(residual)) < self.binning_depth

    def test_residual_diagnostic_flags_within_segment_drift(self):
        # A ramp inside segment 1 violates the constant-shift assumption and
        # must be reported rather than silently absorbed.
        ops = make_ops(nblocks=1, binning_depth=self.binning_depth)
        ops['Nbatches'] = self.n_batches
        shifts = np.zeros(self.n_batches)
        shifts[self.boundary:] = np.linspace(0, 60, self.n_batches - self.boundary)
        st = make_spikes(ops, self.n_batches, shifts, per_batch=800, seed=8)
        seg_of_batch = (np.arange(self.n_batches) >= self.boundary).astype('int64')

        group_id = seg_of_batch[st[:,4].astype('int64')]
        F_seg, _ = datashift.bin_spikes(ops, st, group_id=group_id, n_groups=2)

        with pytest.warns(UserWarning, match='Within-segment drift'):
            ops = datashift.segment_residual_drift(
                ops, st, seg_of_batch, F_seg, device=torch.device('cpu')
                )

        summary = ops['drift_residual_summary']
        spread_seg1 = summary[1,0,3] - summary[1,0,2]
        spread_seg0 = summary[0,0,3] - summary[0,0,2]
        assert spread_seg1 > spread_seg0


# Use `pytest --runslow` option to include these in tests.
@pytest.mark.slow
def test_chronic_pipeline(data_directory, torch_device, capture_mgr, tmp_path):
    """The real pipeline, with the test recording split into 3 fake segments."""
    from kilosort import run_kilosort

    bin_file = data_directory / 'ZFM-02370_mini.imec0.ap.short.bin'
    n_samples = int(io.get_total_samples(bin_file, 385))

    # split into 3 roughly equal segments, deliberately not on batch edges
    starts = [0, n_samples//3, 2*(n_samples//3)]
    seg_file = tmp_path / 'segments.txt'
    seg_file.write_text('\n'.join(str(s) for s in starts))

    with capture_mgr.global_and_fixture_disabled():
        print('\nStarting chronic drift pipeline test...')
        ops, st, clu, _, _, _, _, _, kept = run_kilosort(
            filename=bin_file, device=torch_device,
            settings={'n_chan_bin': 385, 'nblocks': 1,
                      'drift_segment_starts': str(seg_file)},
            probe_name='NeuroPix1_default.mat',
            results_dir=tmp_path / 'results',
            )

    seg = ops['batch_to_segment']
    assert seg is not None
    assert ops['dshift'].shape == (ops['Nbatches'], 2*ops['nblocks'] - 1)
    assert seg.shape == (ops['Nbatches'],)

    # The whole point: the shift is exactly constant within each segment.
    for s in np.unique(seg):
        rows = ops['dshift'][seg == s]
        assert np.all(rows == rows[0]), f'segment {s} shift is not constant'

    # Segments were resolved and the diagnostic ran.
    assert ops['drift_segment_shift'].shape == (np.unique(seg).size,
                                                2*ops['nblocks'] - 1)
    assert 'drift_residual' in ops
    assert (tmp_path / 'results' / 'drift_segments.png').exists()

    # The original path is untouched: settings keep the path we passed in.
    assert ops['settings']['drift_segment_starts'] == str(seg_file)
    assert st.size > 0 and kept.sum() > 0


class TestNoRegression:
    """The default path must be unchanged when no segments are given.

    `bin_spikes` is covered by `TestBinSpikes.test_matches_reference`, which
    compares against the original implementation bit for bit. The only other
    behavioural surface touched in `align_block2` is the new `drift_smoothing`
    argument, which must default to exactly the previous `ops` value.

    """

    def test_default_smoothing_matches_explicit(self):
        ops = make_ops(nblocks=2)
        ops['Nbatches'] = 8
        shifts = np.linspace(0, 15, 8)
        st = make_spikes(ops, 8, shifts, per_batch=300, seed=10)
        F, ysamp = datashift.bin_spikes(ops, st)

        a, yblk_a, _, _ = datashift.align_block2(
            F, ysamp, ops, device=torch.device('cpu')
            )
        b, yblk_b, _, _ = datashift.align_block2(
            F, ysamp, ops, device=torch.device('cpu'),
            drift_smoothing=ops['drift_smoothing']
            )

        assert np.array_equal(a, b)
        assert np.array_equal(yblk_a, yblk_b)

    def test_nbatches_in_ops_is_not_used(self):
        # align_block2 previously read ops['Nbatches']; it must now take the
        # count from F, and give the same answer either way.
        ops = make_ops(nblocks=2)
        ops['Nbatches'] = 8
        st = make_spikes(ops, 8, np.linspace(0, 15, 8), per_batch=300, seed=11)
        F, ysamp = datashift.bin_spikes(ops, st)

        a, _, _, _ = datashift.align_block2(F, ysamp, ops,
                                            device=torch.device('cpu'))
        ops['Nbatches'] = 12345
        b, _, _, _ = datashift.align_block2(F, ysamp, ops,
                                            device=torch.device('cpu'))
        assert np.array_equal(a, b)
