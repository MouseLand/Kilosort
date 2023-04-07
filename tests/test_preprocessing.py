import numpy as np
import scipy.signal as ss
import torch
from torch.fft import fft, ifft, fftshift

import kilosort.preprocessing as kpp
from kilosort import datashift


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


class TestWhitening:
    # Random data to whiten
    n_chans = 100
    x = np.random.rand(n_chans,1000)
    b,a = ss.butter(4, 300, 'high', fs=30000)
    x_filt = ss.filtfilt(b, a, x)
    # Fake x- and y-positions
    xc = np.array([0.0, 1.0]*int(n_chans/2))               
    yc = np.array([np.floor(i/2) for i in range(n_chans)])
    # Add correlation based on distance
    # TODO: This isn't working as expected. Goal is to add correlation between
    #       channels based on distance, e.g. nearby channels should be correlated,
    #       distant channels should not be.
    # TODO: just use sample real data instead.
    # w_corr = np.fromfunction(lambda i,j: np.exp(-(i-j)**2), shape=(n_chans, n_chans))
    x = torch.from_numpy(x.astype('float32')).to('cpu')
    # Get correlation matrix
    cc = (x @ x.T)/x.shape[1]

    def test_whitening_from_covariance(self):
        wm = kpp.whitening_from_covariance(self.cc)
        whitened = (wm @ self.x).numpy()
        new_cov = (whitened @ whitened.T)/whitened.shape[1]

        # Covariance matrix of whitened data should be very close to the
        # identity matrix.
        assert np.allclose(new_cov, np.identity(new_cov.shape[1]), atol=1e-4)

    def test_whitening_local(self):

        wm = kpp.whitening_local(
            self.cc, self.xc, self.yc, nrange=30, device=torch.device('cpu')
            )

        whitened = (wm @ self.x).numpy()
        new_cov = (whitened @ whitened.T)/whitened.shape[1]
        # Covariance matrix of whitened data should be very close to the
        # identity matrix.

        # TODO: Ask Marius about this. I guess it only tries to decorrelate within
        #       the specified nrange? But I'm getting sort of mixed results for
        #       the random data, where some channels are decorrelated with their
        #       neighbors but others aren't.
        # assert np.allclose(new_cov, np.identity(new_cov.shape[1]), atol=1e-4)

    # TODO: This relies on binary file, see notes below about
    #       possibility of disentangling that for easier testing.
    def test_get_whitening(self):
        pass

class TestDriftCorrection:

    def test_datashift(self, bfile, saved_ops, torch_device):
        # TODO: maybe make this a separate test module instead? There's
        #       quite a bit there.
        saved_yblk = saved_ops['yblk']
        saved_dshift = saved_ops['dshift']
        saved_iKxx = saved_ops['iKxx'].to(torch_device)
        ops = datashift.run(saved_ops, bfile, device=torch_device)

        # TODO: this fails on dshift, but the final version doesn't. So, dshift
        #       must be overwritten later on in the pipeline. Need to save the
        #       initial result separately.
        print('testing yblk...')
        assert np.allclose(saved_yblk, ops['yblk'])
        print('testing dshift...')
        assert np.allclose(saved_dshift, ops['dshift'])
        print('testing iKxx...')
        assert torch.allclose(saved_iKxx, ops['iKxx'])
        

    def test_get_drift_matrix(self):
        pass
