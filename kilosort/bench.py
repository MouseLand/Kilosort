import torch, os
from kilosort import preprocessing
dev = torch.device('cuda:0')
import numpy as np
from torch.fft import fft, ifft, fftshift
from scipy.interpolate import interp1d
from tqdm import trange

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
    Kyx = preprocessing.kernel2D_torch(yp, xp, ops['settings']['sig_interp'])
    
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
    NT = ops['batch_size']
    NTbuff   = ops['NTbuff']
    chanMap  = ops['chanMap']
    n_chan_bin = ops['n_chan_bin']
    iKxx = ops['iKxx']
    yblk = ops['yblk']

    with open(filename, mode='rb') as f: 
        # seek the beginning of the batch
        f.seek(2*NT*n_chan_bin*ibatch , 0)

        # go back "NTbuff" samples, unless this is the first batch
        if ibatch==0:
            buff = f.read((NTbuff-nt) * n_chan_bin * 2)
        else:    
            f.seek(- 2*nt*n_chan_bin , 1)
            buff = f.read(NTbuff * n_chan_bin * 2)          

        # read and transpose data
        # this gives a warning, but it's much faster than the alternatives... 
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        data = np.reshape(data, (-1, n_chan_bin)).T

    nsamp = data.shape[-1]
    X = torch.zeros((n_chan_bin, NTbuff), device = dev)

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
        if isinstance(fwav, np.ndarray):
            #fwav = torch.from_numpy(fwav).to(dev)
            hp_filter = preprocessing.get_highpass_filter()
            fwav = preprocessing.fft_highpass(hp_filter, X.shape[1])
        X = torch.real(ifft(fft(X) * torch.conj(fwav)))
        X = fftshift(X, dim = -1)

    # whitening, with optional drift correction
    if Wrot is not None:
        if isinstance(Wrot, np.ndarray):
            Wrot = torch.from_numpy(Wrot).to(dev)
        if dshift is not None:
            if isinstance(iKxx, np.ndarray):
                ops['iKxx'] = torch.from_numpy(iKxx).to(dev)
            M = get_drift_matrix(ops, dshift[ibatch])
            X = (M @ Wrot) @ X
        else:
            X = Wrot @ X

    return X

def avg_wav(filename, Wsub, nn, ops, ibatch, st_i, clu, Nfilt):
    nt = ops['nt']
    NT = ops['batch_size']
    if isinstance(ops['wPCA'], np.ndarray):
        ops['wPCA'] = torch.from_numpy(ops['wPCA']).to(dev)

    X = load_transform(filename, ibatch, ops, fwav = ops['fwav'], 
                                Wrot = ops['Wrot'], dshift = ops['dshift'])

    ix = (st_i - NT * ibatch) * (st_i - NT * (1+ibatch)) < 0
    st_sub = st_i[ix] + nt - NT * ibatch
    st_sub = torch.from_numpy(st_sub).to(dev)

    tiwave = torch.arange(nt, device=dev)
    xsub = X[:, st_sub.unsqueeze(-1) + tiwave] @ ops['wPCA'].T
    nsp = xsub.shape[1]
    M = torch.zeros((Nfilt, nsp), dtype=torch.float, device = dev)
    M[clu[ix], torch.arange(nsp)] = 1
    
    Wsub += torch.einsum('ijk, lj -> lki', xsub, M)
    nn += M.sum(1)

    return Wsub, nn

def clu_ypos(filename, ops, st_i, clu):
    Nfilt = clu.max()+1
    Wsub = torch.zeros((Nfilt, ops['nwaves'], ops['Nchan']), device = dev)
    nn   = torch.zeros((Nfilt, ), device = dev)

    for ibatch in range(0, ops['Nbatches'], 10):
        Wsub, nn = avg_wav(filename, Wsub, nn, ops, ibatch, st_i, clu, Nfilt)

    Wsub = Wsub / nn.unsqueeze(-1).unsqueeze(-1)
    #Wsub = Wsub / nn.unsqueeze(-1).unsqueeze(-1)
    Wsub = Wsub.cpu().numpy()

    ichan = np.argmax((Wsub**2).sum(1), -1)
    yclu = ops['yc'][ichan]

    return yclu, Wsub

def nmatch(ss0, ss, dt=6):
    i = 0
    j = 0
    n0 = 0

    ntmax = len(ss0)
    is_matched  = np.zeros(len(ss), 'bool')
    is_matched0 = np.zeros(len(ss0), 'bool')

    while i<len(ss):
        while j+1<ntmax and ss0[j+1] <= ss[i]+dt:
            j += 1

        if np.abs(ss0[j] - ss[i]) <=dt:
            n0 += 1
            is_matched[i] = 1
            is_matched0[j] = 1

        i+= 1
    return n0, is_matched, is_matched0

def match_neuron(kk, clu, yclu, st_i, clu0, yclu0, st0_i, n_check=20, dt=6):
    ss = st_i[clu==kk]
    isort = np.argsort(np.abs(yclu[kk] - yclu0))
    fmax = 0
    miss = 0
    fpos = 0
    best_ind = isort[0]
    matched_all = -1 * np.ones((n_check,))
    top_inds = -1 * np.ones((n_check,), "int")
    ntest = min(len(isort), n_check)
    top_inds[:ntest] = isort[:ntest]
    for j in range(ntest):
        ss0 = st0_i[clu0==isort[j]]

        if len(ss0) ==0:
            continue
        
        n0, is_matched, is_matched0 = nmatch(ss0, ss, dt=dt)

        #fmax_new = n0 / (len(ss) + len(ss0) - n0)
        fmax_new = 1 - np.maximum(0, 1 - n0/len(ss)) - np.maximum(0, 1 - n0/len(ss0))
        matched_all[j] = n0

        if fmax_new > fmax:
            miss = np.maximum(0, 1 - n0/len(ss))
            fpos = np.maximum(0, 1 - n0/len(ss0))

            fmax = fmax_new
            best_ind = isort[j]

    return fmax, miss, fpos, best_ind, matched_all, top_inds

def compare_recordings(st_gt, clu_gt, yclu_gt, st_new, clu_new, yclu_new):
    NN = len(yclu_gt)

    n_check = 20
    fmax = np.zeros(NN,)
    matched_all = np.zeros((NN, n_check))
    fmiss = np.zeros(NN,)
    fpos = np.zeros(NN,)
    best_ind = np.zeros(NN, "int")
    top_inds = np.zeros((NN, n_check), "int")

    for kk in trange(NN):
        out = match_neuron(kk, clu_gt, yclu_gt, st_gt, clu_new, yclu_new, st_new, n_check=n_check)
        fmax[kk], fmiss[kk], fpos[kk], best_ind[kk], matched_all[kk], top_inds[kk] = out

    return fmax, fmiss, fpos, best_ind, matched_all, top_inds

def load_GT(filename, ops, gt_path, toff = 20, nmax = 600):
    #gt_path = os.path.join(ops['data_folder'] , "sim.imec0.ap_params.npz")
    dd = np.load(gt_path, allow_pickle = True)

    st_gt = dd['st'].astype('int64')
    clu_gt = dd['cl'].astype('int64')

    ix = clu_gt<nmax
    st_gt  = st_gt[ix]
    clu_gt = clu_gt[ix]

    yclu_gt, Wsub = clu_ypos(filename, ops, st_gt - toff, clu_gt.astype('int64'))
    mu_gt = (Wsub**2).sum((1,2))**.5

    unq_clu, nsp = np.unique(clu_gt, return_counts = True)
    if np.abs(np.diff(unq_clu) - 1).sum()>1e-5:
        print('error, some ground truth units are missing')

    return st_gt, clu_gt, yclu_gt, mu_gt, Wsub, nsp


def convert_ks_output(ops, st, clu, toff = 20):
    st = st[:,0].astype('int64')        
    yclu, Wsub    = clu_ypos(ops, st-toff, clu)

    return st, clu, yclu, Wsub


def load_phy(filename, fpath, ops):
    st_new  = np.load(os.path.join(fpath,  "spike_times.npy")).astype('int64')
    try:
        clu_new = np.load(os.path.join(fpath ,"cluster_times.npy")).astype('int64')
    except:
        clu_new = np.load(os.path.join(fpath ,"spike_clusters.npy")).astype('int64')
    if st_new.ndim==2:
        st_new = st_new[:,0]
    if clu_new.ndim==2:
        clu_new = clu_new[:,0]

    yclu_new, Wsub = clu_ypos(filename, ops, st_new - 20, clu_new)

    return st_new, clu_new, yclu_new, Wsub