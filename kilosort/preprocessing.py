import torch, os, scipy
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy import io
from glob import glob
from torch.fft import fft, ifft, fftshift
dev = torch.device('cuda')

def whitening_from_covariance(CC):
    """ whitening matrix for a covariance matrix CC
    """
    E,D,V =  torch.linalg.svd(CC)
    eps = 1e-6
    Wrot =(E / (D+eps)**.5) @ E.T
    return Wrot

def whitening_local(CC, x_chan, y_chan, nrange = 32):
    """ loop through each channel and compute its whitening filter based on nearest channels
    """
    Nchan = CC.shape[0]
    Wrot = torch.zeros((Nchan,Nchan), device = dev)
    for j in range(CC.shape[0]):
        ds = (x_chan[j] - x_chan)**2 + (y_chan[j] - y_chan)**2
        isort = np.argsort(ds)
        ix = isort[:nrange]

        wrot = whitening_from_covariance(CC[np.ix_(ix, ix)])
        Wrot[j, ix] = wrot[0]
    return Wrot

def kernel2D_torch(x, y, sig = 1):
    """ simple Gaussian kernel for two sets of coordinates x and y
    """
    ds = ((x.unsqueeze(1) - y)**2).sum(-1)
    Kn = torch.exp(-ds / (2*sig**2))
    return Kn

def get_drift_matrix(ops, dshift):
    """ for a given dshift drift, computes the linear drift matrix for interpolation
    """

    # first, interpolate drifts to every channel
    yblk = ops['yblk']
    finterp = interp1d(yblk, dshift, fill_value="extrapolate", kind = 'linear')
    shifts = finterp(ops['probe']['yc'])

    # compute coordinates of desired interpolation
    xp = np.vstack((ops['probe']['xc'],ops['probe']['yc'])).T
    yp = xp.copy()
    yp[:,1] -= shifts

    xp = torch.from_numpy(xp).to(dev)
    yp = torch.from_numpy(yp).to(dev)

    # run interpolated to obtain a kernel
    Kyx = kernel2D_torch(yp, xp, ops['settings']['sig_interp'])
    
    # multiply with precomputed kernel matrix of original channels
    M = Kyx @ ops['iKxx']

    return M

"""
def load_transform(filename, ibatch, ops, fwav=None, Wrot = None, dshift = None) :
     this function loads a batch of data ibatch and optionally:
     - if chanMap is present, then the channels are subsampled
     - if fwav is present, then the data is high-pass filtered
     - if Wrot is present,  then the data is whitened
     - if dshift is present, then the data is drift-corrected    
    
    nt = ops['settings']['nt']
    NT = ops['settings']['NT']
    NTbuff   = ops['NTbuff']
    chanMap  = ops['probe']['chanMap']
    NchanTOT = ops['settings']['NchanTOT']

    with io.BinaryFiltered(filename, NchanTOT, NT, nt, 
                            fwav, Wrot, dshift) as f:
        X = f.padded_batch_to_torch(ibatch, ops)
    
    return X
"""
    
    

def get_whitening_matrix(f, x_chan, y_chan, nskip = 25, device=torch.device('cuda')):
    """ get the whitening matrix, use every nskip batches
    """    
    n_chan = len(f.channel_map)
    # collect the covariance matrix across channels
    CC = torch.zeros((n_chan, n_chan), device=device)
    k = 0
    for j in range(0, f.n_batches-1, nskip):
        X = f.padded_batch_to_torch(j, device=device)
        # load data with high-pass filtering
        #X = load_transform(filename, j, ops, fwav = ops['fwav'])
        X = X[:, f.n_twav : -f.n_twav]
        CC = CC + (X @ X.T)/X.shape[1]
        k+=1
        # covariance matrix
    CC = CC / k

    # compute the local whitening filters and collect back into Wrot
    Wrot = whitening_local(CC, x_chan, y_chan)
    return Wrot

def get_highpass_filter(fs = 30000, device=torch.device('cuda')):
    """ filter to use for high-pass filtering. 
    """
    NT = 30122 #500
    b,a = butter(3, 300, fs = fs, btype = 'high')
    x = np.zeros(NT)
    x[NT//2] = 1
    hp_filter = filtfilt(b, a , x).copy()
    hp_filter = torch.from_numpy(hp_filter).to(device).float()
    return hp_filter

def fft_highpass(hp_filter, NT=30122):
    """ convert filter to fourier domain"""
    device = hp_filter.device
    ft = hp_filter.shape[0]
    pad = (NT - ft) // 2
    hp = torch.cat((torch.zeros(pad, device=device), 
                    hp_filter,
                    torch.zeros(pad + (NT-pad*2-ft), device=device)))
    return fft(hp)