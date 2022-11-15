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

def whitening_local(CC, xc, yc, nrange = 32):
    """ loop through each channel and compute its whitening filter based on nearest channels
    """
    Nchan = CC.shape[0]
    Wrot = torch.zeros((Nchan,Nchan), device = dev)
    for j in range(CC.shape[0]):
        ds = (xc[j] - xc)**2 + (yc[j] - yc)**2
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
    shifts = finterp(ops['yc'])

    # compute coordinates of desired interpolation
    xp = np.vstack((ops['xc'],ops['yc'])).T
    yp = xp.copy()
    yp[:,1] -= shifts

    xp = torch.from_numpy(xp).to(dev)
    yp = torch.from_numpy(yp).to(dev)

    # run interpolated to obtain a kernel
    Kyx = kernel2D_torch(yp, xp, ops['sig_interp'])
    
    # multiply with precomputed kernel matrix of original channels
    M = Kyx @ ops['iKxx']

    return M

def load_transform(filename, ibatch, ops, fwav=None, Wrot = None, dshift = None) :
    """ this function loads a batch of data ibatch and optionally:
     - if chanMap is present, then the channels are subsampled
     - if fwav is present, then the data is high-pass filtered
     - if Wrot is present,  then the data is whitened
     - if dshift is present, then the data is drift-corrected    
    """
    nt = ops['nt']
    NT = ops['NT']
    NTbuff   = ops['NTbuff']
    chanMap  = ops['chanMap']
    NchanTOT = ops['NchanTOT']

    with open(filename, mode='rb') as f: 
        # seek the beginning of the batch
        f.seek(2*NT*NchanTOT*ibatch , 0)

        # go back "NTbuff" samples, unless this is the first batch
        if ibatch==0:
            buff = f.read((NTbuff-nt) * NchanTOT * 2)
        else:    
            f.seek(- 2*nt*NchanTOT , 1)
            buff = f.read(NTbuff * NchanTOT * 2)          

        # read and transpose data
        # this gives a warning, but it's much faster than the alternatives... 
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        data = np.reshape(data, (-1, NchanTOT)).T

    nsamp = data.shape[-1]
    X = torch.zeros((NchanTOT, NTbuff), device = dev)

    # fix the data at the edges for the first and last batch
    if ibatch==0:
        X[:, nt:nt+nsamp] = torch.from_numpy(data).to(dev).float()
        X[:, :nt] = X[:, nt:nt+1]
    elif ibatch==ops['Nbatches']-1:
        X[:, :nsamp] = torch.from_numpy(data).to(dev).float()
        X[:, nsamp:] = X[:, nsamp-1:nsamp]
    else:
        X[:] = torch.from_numpy(data).to(dev).float()

    # pick only the channels specified in the chanMap
    if chanMap is not None:
        X = X[chanMap]

    # remove the mean of each channel, and the median across channels
    X = X - X.mean(1).unsqueeze(1)
    X = X - torch.median(X, 0)[0]
  
    # high-pass filtering in the Fourier domain (much faster than filtfilt etc)
    if fwav is not None:
        X = torch.real(ifft(fft(X) * torch.conj(fwav)))
        X = fftshift(X, dim = -1)

    # whitening, with optional drift correction
    if Wrot is not None:
        if dshift is not None:
            M = get_drift_matrix(ops, dshift[ibatch])
            X = (M @ Wrot) @ X
        else:
            X = Wrot @ X

    return X

def find_file(ops):
    ops['filename']  = glob(os.path.join(ops['root_bin'], '*bin'))[0]
    file_size = os.path.getsize(ops['filename'])
    ops['Nbatches'] = (file_size-1)//(ops['NchanTOT']*ops['NT']*2) + 1
    return ops

def get_whitening_matrix(ops, nskip = 25):
    """ get the whitening matrix, use every nskip batches
    """    
    filename = ops['filename']
    Nchan = ops['Nchan']
    Nbatches = ops['Nbatches']
    nt = ops['nt']

    # collect the covariance matrix across channels
    CC = torch.zeros((Nchan,Nchan), device = dev)
    k = 0
    for j in range(0,Nbatches-1, nskip):
        # load data with high-pass filtering
        X = load_transform(filename, j, ops, fwav = ops['fwav'])
        X = X[:,nt:-nt]
        CC = CC + (X @ X.T)/X.shape[1]
        k+=1
        # covariance matrix
    CC = CC / k

    # compute the local whitening filters and collect back into Wrot
    Wrot = whitening_local(CC, ops['xc'], ops['yc'])
    return Wrot

def get_fwav(NT = 30122, fs = 30000):
    """ Fourier filter to use for high-pass filtering. 
    Currently depends on NT, but it could get padded for larger NT. 
    """

    b,a = butter(3, 300, fs = fs, btype = 'high')
    x = np.zeros(NT)
    x[NT//2] = 1
    wav = filtfilt(b,a , x).copy()
    wav = torch.from_numpy(wav).to(dev).float()
    fwav = fft(wav)

    return fwav

def add_config(ops):
    """add configuration options from the matlab probe files
    """

    config_file = os.path.join(ops['root_config'] , ops['config_fname'] )
    dconfig     = io.loadmat(config_file)
    
    connected = dconfig['connected'].astype('bool')

    ops['Nchan']     = connected.sum()

    ops['chanMap']   = dconfig['chanMap'][connected].astype('int16').flatten() - 1
    ops['xc']        = dconfig['xcoords'][connected].astype('float32')
    ops['yc']        = dconfig['ycoords'][connected].astype('float32')

    ops['NTbuff'] = ops['NT'] + 2 * ops['nt']
    return ops

def run(ops):
    # find binary file in the folder
    ops  = find_file(ops)

    # find probe configuration file and load 
    ops  = add_config(ops)

    # precompute high-pass filter
    ops['fwav'] = get_fwav(ops['NTbuff'], ops['fs'])

    # compute whitening matrix
    ops['Wrot'] = get_whitening_matrix(ops, nskip = ops['nskip'])
    return ops