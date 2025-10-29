import os

import torch
import numpy as np
from torch.fft import fft, ifft, fftshift
from scipy.interpolate import interp1d
from tqdm import trange

from kilosort import preprocessing
from kilosort.io import BinaryFiltered


if torch.cuda.is_available():
    dev = torch.device('cuda:0')
else:
    dev = torch.device('cpu')


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


def avg_wav(bf, Wsub, nn, ops, ibatch, st_i, clu, Nfilt, nt, NT):

    if isinstance(ops['wPCA'], np.ndarray):
        ops['wPCA'] = torch.from_numpy(ops['wPCA']).to(dev)


    X = bf.padded_batch_to_torch(ibatch, ops)

    ix = (st_i - bf.imin - NT*ibatch) * (st_i - bf.imin - NT*(1+ibatch)) < 0
    st_sub = st_i[ix] + nt - bf.imin - NT*ibatch
    st_sub = torch.from_numpy(st_sub).to(dev)

    tiwave = torch.arange(nt, device=dev)
    xsub = X[:, st_sub.unsqueeze(-1) + tiwave] @ ops['wPCA'].T
    nsp = xsub.shape[1]
    M = torch.zeros((Nfilt, nsp), dtype=torch.float, device = dev)
    M[clu[ix], torch.arange(nsp)] = 1
    
    Wsub += torch.einsum('ijk, lj -> lki', xsub, M)
    nn += M.sum(1)

    return Wsub, nn


def clu_ypos(filename, ops, st_i, clu, tmin=0.0, tmax=np.inf):
    Nfilt = clu.max()+1
    Wsub = torch.zeros((Nfilt, ops['n_pcs'], ops['Nchan']), device = dev)
    nn   = torch.zeros((Nfilt, ), device = dev)
    fs = ops['fs']
    n_chan_bin = ops['n_chan_bin']
    nt = ops['nt']
    NT = ops['batch_size']
    bf = BinaryFiltered(filename, n_chan_bin=n_chan_bin, fs=fs, NT=NT, nt=nt,
                        hp_filter=ops['fwav'], whiten_mat=ops['Wrot'],
                        dshift=ops['dshift'], chan_map=ops['chanMap'],
                        tmin=tmin, tmax=tmax)

    # Only use batches within tmin, tmax.
    n_batches = min(int((bf.imax-bf.imin)/NT), ops['Nbatches'])
    for ibatch in range(0, n_batches, 10):
        Wsub, nn = avg_wav(bf, Wsub, nn, ops, ibatch, st_i, clu, Nfilt, nt, NT)

    Wsub = Wsub / nn.unsqueeze(-1).unsqueeze(-1)
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
    if len(ss) == 0:
        raise ValueError(f'No GT spikes matching cluster {kk}')
    isort = np.argsort(np.abs(yclu[kk] - yclu0))
    fmax = 0
    miss = 1
    fpos = 0
    best_ind = isort[0]
    matched_all = -1 * np.ones((n_check,))
    top_inds = -1 * np.ones((n_check,), "int")
    ntest = min(len(isort), n_check)
    top_inds[:ntest] = isort[:ntest]

    no_match = 0
    for j in range(ntest):
        ss0 = st0_i[clu0==isort[j]]
        if len(ss0) == 0:
            no_match += 1
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
    if no_match == ntest:
        raise ValueError(f'No matching clusters found for cluster {kk}')

    return fmax, miss, fpos, best_ind, matched_all, top_inds


def compare_recordings(st_gt, clu_gt, yclu_gt, st_new, clu_new, yclu_new):
    NN = len(yclu_gt)

    n_check = 20
    fmax = np.zeros(NN,)
    matched_all = np.zeros((NN, n_check))
    fmiss = np.full(NN, np.nan)
    fpos = np.full(NN, np.nan)
    best_ind = np.zeros(NN, "int")
    top_inds = np.zeros((NN, n_check), "int")

    n_skipped = 0
    for kk in trange(NN):
        try:
            out = match_neuron(kk, clu_gt, yclu_gt, st_gt, clu_new, yclu_new, st_new, n_check=n_check)
            fmax[kk], fmiss[kk], fpos[kk], best_ind[kk], matched_all[kk], top_inds[kk] = out
        except ValueError:
            n_skipped += 1
            continue
    print(f'Num skipped clusters: {n_skipped}')

    return fmax, fmiss, fpos, best_ind, matched_all, top_inds


def load_GT(filename, ops, gt_path, toff=20, nmax=600, tmin=0.0, tmax=np.inf):
    #gt_path = os.path.join(ops['data_folder'] , "sim.imec0.ap_params.npz")
    dd = np.load(gt_path, allow_pickle = True)

    st_gt = dd['st'].astype('int64')
    clu_gt = dd['cl'].astype('int64')

    idx = get_valid_times(st_gt, tmin, tmax, ops['fs'])
    st_gt = st_gt[idx]
    clu_gt = clu_gt[idx]

    ix = clu_gt<nmax
    st_gt  = st_gt[ix]
    clu_gt = clu_gt[ix]

    yclu_gt, Wsub = clu_ypos(filename, ops, st_gt - toff, clu_gt.astype('int64'),
                             tmin=tmin, tmax=tmax)
    mu_gt = (Wsub**2).sum((1,2))**.5

    unq_clu, nsp = np.unique(clu_gt, return_counts = True)
    if np.abs(np.diff(unq_clu) - 1).sum()>1e-5:
        print('error, some ground truth units are missing')

    return st_gt, clu_gt, yclu_gt, mu_gt, Wsub, nsp


def convert_ks_output(ops, st, clu, toff = 20):
    st = st[:,0].astype('int64')        
    yclu, Wsub    = clu_ypos(ops, st-toff, clu)

    return st, clu, yclu, Wsub


def load_phy(filename, fpath, ops, tmin=0.0, tmax=np.inf):
    st_new  = np.load(os.path.join(fpath,  "spike_times.npy")).astype('int64')
    try:
        clu_new = np.load(os.path.join(fpath ,"cluster_times.npy")).astype('int64')
    except:
        clu_new = np.load(os.path.join(fpath ,"spike_clusters.npy")).astype('int64')
    if st_new.ndim==2:
        st_new = st_new[:,0]
    if clu_new.ndim==2:
        clu_new = clu_new[:,0]

    idx = get_valid_times(st_new, tmin, tmax, ops['fs'])
    st_new = st_new[idx]
    clu_new = clu_new[idx]

    yclu_new, Wsub = clu_ypos(filename, ops, st_new - 20, clu_new,
                              tmin=tmin, tmax=tmax)

    return st_new, clu_new, yclu_new, Wsub


def get_valid_times(st, tmin, tmax, fs):
    imin = int(tmin * fs)
    if tmax < np.inf:
        imax = int(tmax * fs)
    else:
        imax = np.inf
    idx = np.logical_and(st >= imin, st < imax)

    return idx
