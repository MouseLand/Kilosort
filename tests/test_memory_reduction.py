"""Checks that the memory-saving rewrites give the same results as before."""

import numpy as np
import torch

from kilosort.utils import BlockBuffer
from kilosort.template_matching import roll_features, permute_rows
from kilosort.postprocessing import compute_spike_positions


def _reference_roll_features(wPCA, tF, Wall, spike_idx, clust_idx, dt):
    """The pre-change implementation (no chunking), kept as a reference."""
    W = wPCA.cpu()
    feats = torch.roll(tF[spike_idx] @ W, shifts=dt, dims=2)
    temps = torch.roll(Wall[clust_idx:clust_idx+1] @ wPCA, shifts=dt, dims=2)
    if dt > 0:
        feats[:,:,:dt] = feats[:,:,dt].unsqueeze(-1)
        temps[:,:,:dt] = temps[:,:,dt].unsqueeze(-1)
    elif dt < 0:
        feats[:,:,dt:] = feats[:,:,dt-1].unsqueeze(-1)
        temps[:,:,dt:] = temps[:,:,dt-1].unsqueeze(-1)
    tF[spike_idx] = feats @ W.T
    Wall[clust_idx] = temps @ wPCA.T
    return tF, Wall


class TestBlockBuffer:
    def _chunks(self):
        rng = np.random.default_rng(0)
        return [rng.normal(size=(n, 4, 3)).astype('float32')
                for n in [5, 0, 1, 17, 3, 40]]

    def test_matches_np_concatenate(self):
        chunks = self._chunks()
        expected = np.concatenate(chunks, 0)
        # block of 7 rows, so chunks span several blocks
        buf = BlockBuffer((4, 3), 'float32', block_bytes=7*4*3*4)
        for c in chunks:
            buf.append(c)
        assert len(buf) == expected.shape[0]
        assert len(buf.blocks) > 1
        out = buf.finalize()
        assert out.dtype == np.float32
        assert np.array_equal(out, expected)
        assert len(buf) == 0 and buf.blocks == []

    def test_dest_index_equals_reindexing(self):
        chunks = self._chunks()
        full = np.concatenate(chunks, 0)
        isort = np.random.default_rng(2).permutation(full.shape[0])
        inv = np.empty_like(isort)
        inv[isort] = np.arange(isort.size)
        buf = BlockBuffer((4, 3), 'float32', block_bytes=7*4*3*4)
        for c in chunks:
            buf.append(c)
        assert np.array_equal(buf.finalize(dest_index=inv), full[isort])

    def test_torch_input_and_cast(self):
        x = torch.randn(9, 2)
        buf = BlockBuffer((2,), 'float64', block_bytes=4*2*8)
        buf.append(x)
        buf.append(x.T.contiguous().T)   # non-contiguous view
        out = buf.finalize()
        assert out.dtype == np.float64
        assert np.array_equal(out, np.concatenate([x.numpy(), x.numpy()]).astype('float64'))

    def test_empty(self):
        out = BlockBuffer((6,), 'float64').finalize()
        assert out.shape == (0, 6)
        assert out.dtype == np.float64


class TestRollFeatures:
    def test_chunked_matches_reference(self):
        torch.manual_seed(0)
        n_spikes, nC, n_pcs, nt, n_clu = 1000, 10, 6, 61, 5
        wPCA = torch.randn(n_pcs, nt)
        clu = np.random.default_rng(1).integers(0, n_clu, n_spikes)
        for dt in [-3, -1, 1, 4]:
            tF = torch.randn(n_spikes, nC, n_pcs)
            Wall = torch.randn(n_clu, 20, n_pcs)
            idx = clu == 2
            tF_ref, Wall_ref = _reference_roll_features(
                wPCA, tF.clone(), Wall.clone(), idx, 2, dt
                )
            for chunk_size in [1, 7, 2**16]:
                tF_new, Wall_new = roll_features(
                    wPCA, tF.clone(), Wall.clone(), idx, 2, dt,
                    chunk_size=chunk_size
                    )
                assert torch.equal(tF_new, tF_ref)
                assert torch.equal(Wall_new, Wall_ref)


class TestPermuteRows:
    def _orders(self, n):
        rng = np.random.default_rng(3)
        near = np.argsort(np.arange(n) + rng.integers(-3, 4, n), kind='stable')
        return [rng.permutation(n), near, np.arange(n), np.arange(n)[::-1].copy()]

    def test_numpy_in_place(self):
        st = np.random.default_rng(4).normal(size=(500, 3))
        for order in self._orders(500):
            x = st.copy()
            out = permute_rows(x, order)
            assert out is x
            assert np.array_equal(out, st[order])

    def test_torch_in_place(self):
        tF = torch.randn(500, 10, 6)
        for order in self._orders(500):
            x = tF.clone()
            out = permute_rows(x, order)
            assert out.data_ptr() == x.data_ptr()
            assert torch.equal(out, tF[torch.from_numpy(order)])

    def test_non_contiguous_falls_back_to_copy(self):
        base = torch.randn(6, 50, 4)
        x = base.transpose(0, 1)   # (50, 6, 4), not contiguous
        before = x.clone()
        order = np.random.default_rng(5).permutation(50)
        out = permute_rows(x, order)
        assert torch.equal(out, before[torch.from_numpy(order)])
        assert torch.equal(x, before)

    def test_empty(self):
        assert permute_rows(np.zeros((0, 3)), np.zeros(0, int)).shape == (0, 3)


def _reference_spike_positions(st, tF, ops):
    """The pre-change compute_spike_positions (no chunking)."""
    cpu = torch.device('cpu')
    icc_mask = ops['iCC_mask'].to(cpu)
    iU = ops['iU'].to(cpu)
    iCC = ops['iCC'].to(cpu)
    tmass = torch.norm(tF, 2, dim=-1)
    tmask = icc_mask[:, iU[st[:,1]]].T
    tmass = tmass * tmask
    tmass = tmass / tmass.sum(1, keepdim=True)
    xc = torch.from_numpy(ops['xc'])
    yc = torch.from_numpy(ops['yc'])
    chs = iCC[:, iU[st[:,1]]]
    xs = (xc[chs.T] * tmass).sum(1).numpy()
    ys = (yc[chs.T] * tmass).sum(1).numpy()
    return xs, ys


class TestSpikePositions:
    def test_chunked_matches_reference(self):
        rng = np.random.default_rng(6)
        n_chan, nC, n_templates, n_spikes = 32, 10, 40, 5001
        ops = {
            'xc': rng.uniform(0, 40, n_chan).astype('float32'),
            'yc': rng.uniform(0, 600, n_chan).astype('float32'),
            'iCC': torch.from_numpy(rng.integers(0, n_chan, (nC, n_chan))),
            'iCC_mask': torch.from_numpy(rng.random((nC, n_chan)) < 0.7),
            'iU': torch.from_numpy(rng.integers(0, n_chan, n_templates)),
            }
        st = np.zeros((n_spikes, 3))
        st[:, 0] = np.sort(rng.integers(0, 10**6, n_spikes))
        st[:, 1] = rng.integers(0, n_templates, n_spikes)
        tF = torch.randn(n_spikes, nC, 6)
        xs_ref, ys_ref = _reference_spike_positions(st, tF, ops)
        # odd sizes are rounded up to a multiple of 1024 internally
        for chunk_size in [1, 1000, 1024, 4096, 2**18]:
            xs, ys = compute_spike_positions(st, tF, ops, chunk_size=chunk_size)
            assert xs.dtype == xs_ref.dtype and xs.shape == xs_ref.shape
            # NaN where every channel is masked, same as before
            assert np.array_equal(xs, xs_ref, equal_nan=True)
            assert np.array_equal(ys, ys_ref, equal_nan=True)


def _fake_make_pc_features(ops, spike_templates, spike_clusters, tF):
    """Stand-in with the same in-place access pattern as make_pc_features:
    per cluster, read some rows (possibly already modified) and write them."""
    ind = []
    for i in np.unique(spike_clusters):
        rows = torch.from_numpy(np.flatnonzero(spike_templates % 3 == i % 3))
        tF[rows, :] = tF[rows].flip(1) * 0.5 + float(i)
        ind.append(np.arange(tF.shape[1]) + i)
    return torch.permute(tF, (0, 2, 1)), np.array(ind, dtype=np.uint32)


class TestSavePcFeatures:
    def test_same_file_as_np_save(self, tmp_path, monkeypatch):
        import kilosort.io as kio
        monkeypatch.setattr(kio, 'make_pc_features', _fake_make_pc_features)
        rng = np.random.default_rng(7)
        n_spikes = 2000
        tF = torch.randn(n_spikes, 10, 6)
        tF_before = tF.clone()
        kept = rng.random(n_spikes) < 0.9
        templates = rng.integers(0, 30, kept.sum()).astype('int32')
        clusters = rng.integers(0, 12, kept.sum()).astype('int32')

        # Old code path
        pc_ref, ind_ref = _fake_make_pc_features(None, templates, clusters,
                                                 tF[kept])
        np.save(tmp_path / 'ref.npy', pc_ref)

        for chunk_size in [7, 2**18]:
            ind = kio.save_pc_features(
                tmp_path / 'new.npy', None, templates, clusters, tF, kept,
                chunk_size=chunk_size
                )
            assert np.array_equal(ind, ind_ref)
            assert ((tmp_path / 'new.npy').read_bytes()
                    == (tmp_path / 'ref.npy').read_bytes())
            assert torch.equal(tF, tF_before)
