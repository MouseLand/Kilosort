import kilosort.preprocessing as kpp
import numpy as np
import torch
from torch.fft import fft, ifft, fftshift

np.random.seed(123)
# TODO: Preprocessing unit tests, group by pipeline step

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

        # TODO: are there specific properties of the filter that should be checked?
        #       the function is pretty much just a couple scipy calls, so I'm not
        #       sure what else I should check here that wouldn't be trivial.

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
    x = torch.from_numpy(np.random.rand(10,1000).astype('float32'))
    cc = (x @ x.T)/x.shape[1]

    def test_whitening_from_covariance(self):
        wm = kpp.whitening_from_covariance(self.cc)
        whitened = (wm @ self.x).numpy()
        new_cov = (whitened @ whitened.T)/whitened.shape[1]

        # Covariance matrix of whitened data should be very close to the
        # identity matrix.
        assert np.allclose(new_cov, np.identity(new_cov.shape[1]), atol=1e-4)

    def test_whitening_local(self):
        xc = np.array([0.0, 1.0]*5)                        # fake x-positions
        yc = np.array([np.floor(i/2) for i in range(10)])  # fake y-positions
        wm = kpp.whitening_local(
            self.cc, xc, yc, nrange=3, device=torch.device('cpu')
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
    pass


# TODO: in separate test module, start on rough determinism/exact match test,
#       e.g. break up into steps as much as possible and check that output
#       matches pre-computed version after each step. Need to slice out a reasonable
#       size chunk of sample data to do this with, so that each step can be
#       saved to file without taking up a lot of space.
# TODO: How to set up a binary file for this, to do the full pipeline?
# TODO: Separately, is there some way we can disentangle the binary file read/write
#       from the other code, for easier testing? Ultimately, all of the operations
#       are actually applied to some tensor in pytorch, so ideally the main
#       data wrapper would just store that (with some separate object referenced
#       for updating the current tensor).

# NOTE: Any changes there have to be mindful of the output format expected by
#       Phy and SpikeInterface