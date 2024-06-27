import torch, os, scipy
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from glob import glob
from torch.fft import fft, ifft, fftshift

def whitening_from_covariance(CC):
    """Whitening matrix for a covariance matrix CC.

    This is the so-called ZCA whitening matrix.

    """
    E,D,V =  torch.linalg.svd(CC)
    eps = 1e-6
    Wrot =(E / (D+eps)**.5) @ E.T
    return Wrot

def whitening_local(CC, xc, yc, nrange=32, device=torch.device('cuda')):
    """Compute whitening filter for each channel based on nearest channels."""
    Nchan = CC.shape[0]
    Wrot = torch.zeros((Nchan,Nchan), device = device)

    # for each channel, a local covariance matrix is extracted
    # the whitening matrix is computed for that local neighborhood
    for j in range(CC.shape[0]):
        ds = (xc[j] - xc)**2 + (yc[j] - yc)**2
        isort = np.argsort(ds)
        ix = isort[:nrange]

        wrot = whitening_from_covariance(CC[np.ix_(ix, ix)])

        # the first row of wrot is a whitening vector for the center channel
        Wrot[j, ix] = wrot[0]
    return Wrot

def kernel2D_torch(x, y, sig = 1):
    """Simple Gaussian kernel for two sets of coordinates x and y."""
    ds = ((x.unsqueeze(1) - y)**2).sum(-1)
    Kn = torch.exp(-ds / (2*sig**2))
    return Kn


def get_drift_matrix(ops, dshift, device=torch.device('cuda')):
    """For given dshift drift, compute linear drift matrix for interpolation."""

    # first, interpolate drifts to every channel
    yblk = ops['yblk']
    if ops['nblocks'] == 1:
        shifts = dshift
    else:
        finterp = interp1d(yblk, dshift, fill_value="extrapolate", kind = 'linear')
        shifts = finterp(ops['probe']['yc'])

    # compute coordinates of desired interpolation
    xp = np.vstack((ops['probe']['xc'],ops['probe']['yc'])).T
    yp = xp.copy()
    yp[:,1] -= shifts

    xp = torch.from_numpy(xp).to(device)
    yp = torch.from_numpy(yp).to(device)

    # the kernel is radial symmetric based on distance
    Kyx = kernel2D_torch(yp, xp, ops['settings']['sig_interp'])
    
    # multiply with precomputed inverse kernel matrix of original channels
    M = Kyx @ ops['iKxx']

    return M


def get_fwav(NT = 30122, fs = 30000, device=torch.device('cuda')):
    """Precomputes a filter to use for high-pass filtering.
    
    To be used with fft in pytorch. Currently depends on NT,
    but it could get padded for larger NT.

    """

    # a butterworth filter is specified in scipy
    b,a = butter(3, 300, fs = fs, btype = 'high')
    
    # a signal with a single entry is used to compute the impulse response
    x = np.zeros(NT)
    x[NT//2] = 1
    
    # symmetric filter from scipy
    wav = filtfilt(b,a , x).copy()
    wav = torch.from_numpy(wav).to(device).float()

    # the filter will be used directly in the Fourier domain
    fwav = fft(wav)

    return fwav

def get_whitening_matrix(f, xc, yc, nskip=25, nrange=32):
    """Get the whitening matrix, use every nskip batches."""
    n_chan = len(f.chan_map)
    # collect the covariance matrix across channels
    CC = torch.zeros((n_chan, n_chan), device=f.device)
    k = 0
    for j in range(0, f.n_batches-1, nskip):
        # load data with high-pass filtering (see the Binary file class)
        X = f.padded_batch_to_torch(j)        
        
        # remove padding
        X = X[:, f.nt : -f.nt]

        # cumulative covariance matrix
        CC = CC + (X @ X.T)/X.shape[1]
        
        k+=1
        
    CC = CC / k

    # compute the local whitening filters and collect back into Wrot
    Wrot = whitening_local(CC, xc, yc, nrange=nrange, device=f.device)

    return Wrot

def get_highpass_filter(fs=30000, cutoff=300, device=torch.device('cuda')):
    """Filter to use for high-pass filtering."""
    NT = 30122
    
    # a butterworth filter is specified in scipy
    b,a = butter(3, cutoff, fs=fs, btype='high')

    # a signal with a single entry is used to compute the impulse response
    x = np.zeros(NT)
    x[NT//2] = 1

    # symmetric filter from scipy
    hp_filter = filtfilt(b, a , x).copy()
    
    hp_filter = torch.from_numpy(hp_filter).to(device).float()
    return hp_filter

def fft_highpass(hp_filter, NT=30122):
    """Convert filter to fourier domain."""
    device = hp_filter.device
    ft = hp_filter.shape[0]

    # the filter is padded or cropped depending on the size of NT
    if ft < NT:
        pad = (NT - ft) // 2
        fhp = fft(torch.cat((torch.zeros(pad, device=device), 
                             hp_filter,
                             torch.zeros(pad + (NT-pad*2-ft), device=device))))
    elif ft > NT:
        crop = (ft - NT) // 2 
        fhp = fft(hp_filter[crop : crop + NT])
    else:
        fhp = fft(hp_filter)
    return fhp
