from scipy.sparse import coo_matrix
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from kilosort import spikedetect

def bin_spikes(ops, st):
    ymin = ops['yc'].min()
    ymax = ops['yc'].max()
    dd = ops['binning_depth'] # binning width in depth
    dmin = ymin-1
    dmax = 1 + np.ceil((ymax-dmin)/dd).astype('int32')
    Nbatches = ops['Nbatches']

    batch_id = st[:,4].copy()

    F = np.zeros((Nbatches, dmax, 20))
    for t in range(ops['Nbatches']):
        ix = (batch_id==t).nonzero()[0]
        sst = st[ix]

        dep = sst[:,1] - dmin
        amp = np.log10(np.minimum(99, sst[:,2])) - np.log10(ops['Th_detect'])
        amp = amp / (np.log10(100)-np.log10(ops['Th_detect']))

        rows = (dep/dd).astype('int32')
        cols = (1e-5 + amp * 20).astype('int32')
        cou = np.ones(len(ix))

        M = coo_matrix((cou, (rows, cols)), (dmax, 20))
        F[t] = np.log2(1+M.todense())

    ysamp = dmin + dd * np.arange(dmax) - dd/2
    return F, ysamp


def align_block2(F, ysamp, ops, device=torch.device('cuda')):

    Nbatches = ops['Nbatches']
    n = 15
    dc = np.zeros((2*n+1, Nbatches))
    dt = np.arange(-n,n+1,1)
    Fg = torch.from_numpy(F).to(device).float()
    Fg = Fg - Fg.mean(1).unsqueeze(1)
    F0 = Fg[np.minimum(300, Nbatches//2)]


    niter = 10
    dall = np.zeros((niter, Nbatches))

    for iter in range(niter):
        for t in range(len(dt)):
            Fs = torch.roll(Fg, dt[t], 1)
            dc[t] = (Fs * F0).mean(-1).mean(-1).cpu().numpy()

        if iter<niter-1:
            imax = np.argmax(dc, 0)

            for t in range(len(dt)):
                ib = imax==t
                Fg[ib] = torch.roll(Fg[ib], dt[t], 1)
                dall[iter, ib] = dt[t]
        F0 = Fg.mean(0)

    nblocks = ops['nblocks']
    nybins = F.shape[1]
    yl = nybins//nblocks
    ifirst = np.round(np.linspace(0,nybins-yl, 2 *nblocks-1)).astype('int32')
    ilast = ifirst + yl

    nblocks = len(ifirst)
    yblk = np.zeros(nblocks,)
    n  = 5
    dt = np.arange(-n, n+1, 1)
    dcs = np.zeros((2*n+1, Nbatches, nblocks))

    for j in range(nblocks):
        isub = np.arange(ifirst[j], ilast[j], 1)
        yblk[j] = ysamp[isub].mean()

        Fsub = Fg[:, isub]

        for t in range(len(dt)):
            Fs = torch.roll(Fsub, dt[t], 1)
            dcs[t, :, j] = (Fs * F0[isub]).mean(-1).mean(-1).cpu().numpy()

    dtup = np.linspace(-n,n,2*n*10+1)
    Kn = kernelD(dt,dtup,1) 
    dcs = gaussian_filter(dcs, [0.5, 0.5, 0.5])

    imin = np.zeros((Nbatches, nblocks))
    for j in range(nblocks):
        dcup = Kn.T @ dcs[:,:,j]
        imax = np.argmax(dcup, 0)
        dall[niter-1] = dtup[imax]
        imin[:,j] = dall.sum(0)

    Fg = torch.from_numpy(F).float()
    imax = dall[:niter-1].sum(0)

    for t in range(len(dt)):
        ib = imax==dt[t]
        Fg[ib] = torch.roll(Fg[ib], dt[t], 1)
    F0m = Fg.mean(0)

    return imin, yblk, F0, F0m


def kernelD(x, y, sig = 1):
    ds = (x[:,np.newaxis] - y)
    Kn = np.exp(-ds**2 / (2*sig**2))
    return Kn
    
def kernel2D_torch(x, y, sig = 1):
    ds = ((x.unsqueeze(1) - y)**2).sum(-1)
    Kn = torch.exp(-ds / (2*sig**2))
    return Kn

def kernel2D(x, y, sig = 1):
    ds = ((x[:,np.newaxis] - y)**2).sum(-1)
    Kn = np.exp(-ds / (2*sig**2))
    return Kn

def run(ops, device=torch.device('cuda')):
    st, tF, ops  = spikedetect.run(ops, device=device)
    F, ysamp = bin_spikes(ops, st)
    imin, yblk, F0, F0m = align_block2(F, ysamp, ops, device=device)

    dshift = imin * ops['binning_depth']
    ops['yblk'] = yblk
    ops['dshift'] = dshift 
    xp = np.vstack((ops['xc'],ops['yc'])).T
    Kxx = kernel2D(xp, xp, ops['sig_interp'])
    Kxx = torch.from_numpy(Kxx).to(device)
    ops['iKxx'] = torch.linalg.inv(Kxx + 0.01 * torch.eye(Kxx.shape[0], device=device))
    return ops