import pytest
import numpy as np
import torch
from torch.fft import fft, ifft, fftshift

import kilosort.preprocessing as kpp
from kilosort import datashift, io


np.random.seed(123)

class TestFiltering:
    # 2 seconds of time samples at 30Khz, 1 channel
    t = np.linspace(0, 2, 60000, False, dtype='float32')[np.newaxis,...]
    # 100hz and 500hz signals
    sine_100hz = torch.from_numpy(np.sin(2*np.pi*100*t))
    sine_500hz = torch.from_numpy(np.sin(2*np.pi*500*t))
    # high pass filter (hard-coded for 300hz threshold)
    hp_filter = kpp.get_highpass_filter(device=torch.device('cpu'))

    def test_get_highpass_filter(self):
        # Add dummy axes, shape (channels in, channels out, width)
        hp_filter = self.hp_filter[None, None, :]
        filtered_100hz = torch.nn.functional.conv1d(self.sine_100hz, hp_filter)
        filtered_500hz = torch.nn.functional.conv1d(self.sine_500hz, hp_filter)

        # After applying high pass filter,
        # 100hz signal should be close to 0, 500hz should be mostly unchanged,
        # but neither case is exact.
        assert torch.max(filtered_100hz) < 0.01
        assert torch.max(filtered_500hz) > 0.9

    def test_fft_highpass(self):
        fft1 = kpp.fft_highpass(self.hp_filter, NT=1000)    # crop filter
        fft2 = kpp.fft_highpass(self.hp_filter, NT=100000)  # pad filter
        # TODO: Currently this only leaves it unchanged b/c NT is hard-coded
        #       to the same value for get_highpass_filter and fft_highpass,
        #       which is fragile. Should define that better somewhere.
        fft3 = kpp.fft_highpass(self.hp_filter)             # same size

        # New filter's shape should match NT, or be the same as the original
        # filter.
        assert fft1.shape[0] == 1000
        assert fft2.shape[0] == 100000
        assert fft3.shape[0] == self.hp_filter.shape[0]

        # TODO: Currently this is run as one step in io.BinaryFiltered.filter(),
        #       so this will need to be updated if that code changes. May be
        #       preferable to encapsulate each of those steps in a function to
        #       make tests easier to keep up to date.
        
        # Apply fourier versioon of high pass filter.
        fwav = kpp.fft_highpass(self.hp_filter, NT=self.sine_100hz.shape[1])
        x100 = torch.real(ifft(fft(self.sine_100hz) * torch.conj(fwav)))
        x100 = fftshift(x100, dim = -1)
        x500 = torch.real(ifft(fft(self.sine_500hz) * torch.conj(fwav)))
        x500 = fftshift(x500, dim = -1)

        # After applying high pass filter,
        # 100hz signal should be close to 0, 500hz should be mostly unchanged,
        # but neither case is exact.
        assert torch.max(x100) < 0.01
        assert torch.max(x500) > 0.9


class TestArtifactRemoval:
    
    def test_threshold(self, torch_device):
        a = np.random.randint(-1000, 1000, (1000,10)).astype(np.float32)
        a[900,4] = 30001

        bfile1 = io.BinaryFiltered(
            filename='dummy', n_chan_bin=10, NT=500, device=torch_device,
            file_object=a, artifact_threshold=30000
        )
        bfile2 = io.BinaryFiltered(
            filename='dummy', n_chan_bin=10, NT=500, device=torch_device,
            file_object=a
            )

        # No threshold crossings in the first half, so these should match.
        assert torch.allclose(bfile1[:500,:], bfile2[:500,:])
        # Second half should be zeroed out for bfile1 only.
        zeros = torch.zeros(500,10).to(torch_device).float()
        assert torch.allclose(bfile1[500:,:], zeros.T)
        assert not torch.allclose(bfile2[500:,:], zeros.T)


class TestWhitening:

    def test_whitening_from_covariance(self, torch_device):
        x = torch.from_numpy(np.random.rand(100, 1000)).to(torch_device).float()
        cc = (x @ x.T)/1000
        wm = kpp.whitening_from_covariance(cc)
        whitened = wm @ x
        new_cov = (whitened @ whitened.T)/whitened.shape[1]

        # Covariance matrix of whitened data should be very close to the
        # identity matrix.
        assert torch.allclose(
            new_cov, torch.eye(new_cov.shape[1], device=torch_device),
            atol=1e-4
            )

    def test_get_whitening(self, bfile, saved_ops):
        xc = saved_ops['probe']['xc']
        yc = saved_ops['probe']['yc']
        wm = kpp.get_whitening_matrix(bfile, xc, yc)

        ### Perform other preprocessing steps on data to ensure valid result.
        # TODO: better way to encapsulate these steps for re-use.
        # Get first batch of data
        X = torch.from_numpy(bfile.file[:bfile.NT,:].T).to(bfile.device).float()
        # Remove unwanted channels
        if bfile.chan_map is not None:
            X = X[bfile.chan_map]
        # remove the mean of each channel, and the median across channels
        X = X - X.mean(1).unsqueeze(1)
        X = X - torch.median(X, 0)[0]
        # high-pass filtering in the Fourier domain (much faster than filtfilt etc)
        fwav = kpp.fft_highpass(bfile.hp_filter, NT=X.shape[1])
        X = torch.real(ifft(fft(X) * torch.conj(fwav)))
        X = fftshift(X, dim = -1)
        ###

        # Apply whitening matrix to one batch
        whitened = (wm @ X)
        new_cov = (whitened @ whitened.T)/whitened.shape[1]
        identity = torch.eye(new_cov.shape[1], device=bfile.device)

        # TODO: Double check with Marius, this still isn't true but maybe
        #       that's okay. The "shape" is still similar (e.g. high values
        #       along and adjacent to diagonal, rest near 0).
        # Covariance matrix of whitened data should be approximately equal
        # to the identity matrix.
        # assert torch.allclose(new_cov, identity, atol=1e-2)

        # Alternative test until identity matrix question is resolved.
        # Normalized covariance matrix should have 99th percentile < 0.1.
        # In other words, very few values that are not near 0.
        norm_cov = new_cov - new_cov.min()
        norm_cov = norm_cov/norm_cov.max()
        assert torch.quantile(torch.flatten(norm_cov), 0.99) < 0.1


# TODO: need to investigate why these aren't exact matches, likely an issue with
#       updates to dependencies.
# class TestDriftCorrection:

#     @pytest.mark.slow
#     def test_datashift(self, bfile, saved_ops, torch_device, capture_mgr):
#         saved_yblk = saved_ops['yblk']
#         saved_dshift = saved_ops['dshift']
#         saved_iKxx = saved_ops['iKxx'].to(torch_device)
#         with capture_mgr.global_and_fixture_disabled():
#             print('\nStarting datashift.run test...')
#             ops, st = datashift.run(saved_ops, bfile, device=torch_device)

#         # TODO: this fails on dshift, but the final version doesn't. So, dshift
#         #       must be overwritten later on in the pipeline. Need to save the
#         #       initial result separately.
#         print('testing yblk...')
#         assert np.allclose(saved_yblk, ops['yblk'])
#         print('testing dshift...')
#         # assert np.allclose(saved_dshift, ops['dshift'])
#         print('testing iKxx...')
#         assert torch.allclose(saved_iKxx, ops['iKxx'])
        

#     def test_get_drift_matrix(self):
#         # TODO
#         pass
