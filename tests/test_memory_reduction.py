"""Checks that the memory-saving rewrites give the same results as before."""

import numpy as np
import torch

from kilosort.utils import BlockBuffer
from kilosort.template_matching import roll_features


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

