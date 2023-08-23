from io import StringIO
import time
from torch.nn.functional import max_pool2d, avg_pool2d, conv1d, max_pool1d
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from kilosort.utils import template_path

dev = torch.device('cuda')

def find_peaks(X, ops, loc_range = [5,5], long_range = [7,31]):
    nt = ops['nt']
    Th = ops['Th']
    Xneg = -X.unsqueeze(0)
    Xmax = max_pool2d(Xneg, (loc_range[0], 1), stride=1, padding=(loc_range[0]//2, 0))
    Xmax = max_pool2d(Xmax, (1, loc_range[1]), stride=1, padding=(0, loc_range[1]//2))

    peaks = torch.logical_and(Xmax == Xneg, Xmax > Th).float()
    peaks[0, :,:nt] = 0
    peaks[0, :,-nt:] = 0

    Pmax = avg_pool2d(peaks, (long_range[0], 1), stride=1, padding=(long_range[0]//2, 0))
    Pmax = avg_pool2d(Pmax, (1, long_range[1]), stride=1, padding=(0, long_range[1]//2))
    Pmax = Pmax * np.prod(long_range)

    is_peak = torch.logical_and(peaks[0], Pmax[0] < 1.2)
    xs = torch.nonzero(is_peak)
    return xs

def get_waves(ops, device=torch.device('cuda')):
    dd = np.load(template_path())
    wTEMP = torch.from_numpy(dd['wTEMP']).to(device)
    wPCA = torch.from_numpy(dd['wPCA']).to(device)
    return wPCA, wTEMP

def template_centers(ops):
    xmin, xmax, ymin, ymax = ops['xc'].min(), ops['xc'].max(), ops['yc'].min(), ops['yc'].max()
    dmin = np.median(np.diff(np.unique(ops['yc'])))
    ops['yup'] = np.arange(ymin, ymax+.00001, dmin//2)
    ops['dmin'] = dmin 

    yunq = np.unique(ops['yc'])
    mxc = np.NaN * np.ones(len(yunq))
    for j in range(len(yunq)):
        xc = ops['xc'][ops['yc']==yunq[j]]
        if len(xc)>1:
            mxc[j] = np.median(np.diff(np.sort(xc)))
        else:
            mxc[j] = 0
    dminx = np.nanmedian(mxc)
    ops['dminx'] = dminx
    if dminx>0:
        nx = np.round((xmax - xmin) / (dminx/2)) + 1
    else:
        dmix = 10
        nx = 1
    ops['xup'] = np.linspace(xmin, xmax, int(nx))
    return ops


def template_match(X, ops, iC, iC2, weigh, device=torch.device('cuda')):
    NT = X.shape[-1]
    nt = ops['nt']
    Nchan = ops['Nchan']
    Nfilt = iC.shape[1]

    tch0 = torch.zeros(1,device = device)
    tch1 = torch.ones(1,device = device)

    W = ops['wTEMP'].unsqueeze(1)
    B = conv1d(X.unsqueeze(1), W, padding=nt//2)

    nt0 = 20
    nk = ops['wTEMP'].shape[0] #ops['nwaves']

    niter = 40
    nb = (NT-1)//niter+1
    As    = torch.zeros((Nfilt, NT), device=device)
    Amaxs = torch.zeros((Nfilt, NT), device=device)
    imaxs = torch.zeros((Nfilt, NT), dtype = torch.int64, device=device)

    ti = torch.arange(Nfilt, device = device)
    tj = torch.arange(nb, device = device)

    for t in range(niter):
        A = torch.einsum('ijk, jklm-> iklm', weigh, B[iC,:, nb*t:nb*(t+1)])        
        A = A.transpose(1,2)
        A = A.reshape(-1, Nfilt, A.shape[-1])
        
        #Aa, imax = torch.max(A, 0) 
        Aa, imax = torch.max(A.abs(), 0)
        imax = (1+imax) * A[imax, ti.unsqueeze(-1), tj[:A.shape[-1]]].sign()

        As[:, nb*t:nb*(t+1)] = Aa
        imaxs[:, nb*t:nb*(t+1)] = imax
        Amax = torch.max(Aa[iC2], 0)[0]
        Amaxs[:, nb*t:nb*(t+1)] = Amax

    Amaxs[:,:nt] = 0
    Amaxs[:,-nt:] = 0
    Amaxs  = max_pool1d(Amaxs.unsqueeze(0), (2*nt0+1), stride = 1, padding = nt0).squeeze(0)
    xy = torch.logical_and(Amaxs==As, As > ops['Th_detect']).nonzero()
    imax = imaxs[xy[:,0], xy[:,1]]
    amp = As[xy[:,0], xy[:,1]]

    ssign = imax.sign()
    imax = imax.abs()-1
    adist = B[iC[:, xy[:,0]], imax%nk, xy[:,1]] * ssign

    #adist = B[iC[:, xy[:,0]], imax%nk, xy[:,1]] 
    
    #xy[:,1] -= nt
    return xy, imax, amp, adist


def nearest_chans(ys, yc, xs, xc, nC, device=torch.device('cuda')):
    ds = (ys - yc[:,np.newaxis])**2 + (xs - xc[:,np.newaxis])**2
    iC = np.argsort(ds, 0)[:nC]
    iC = torch.from_numpy(iC).to(device)
    ds = np.sort(ds, 0)[:nC]
    return iC, ds

def yweighted(yc, iC, adist, xy, device=torch.device('cuda')):    

    yy = torch.from_numpy(yc).to(device)[iC]
    cF0 = torch.nn.functional.relu(adist)
    cF0 = cF0/cF0.sum(0)

    yct = (cF0 * yy[:,xy[:,0]]).sum(0)
    return yct

def run(ops, bfile, device=torch.device('cuda'), progress_bar=None):        
    sig = 10
    nsizes = 5

    ops['wPCA'], ops['wTEMP'] = get_waves(ops, device=device)
    #print(ops['wTEMP'].shape, ops['wPCA'].shape)

    ops = template_centers(ops)   

    [ys, xs] = np.meshgrid(ops['yup'], ops['xup'])
    ys, xs = ys.flatten(), xs.flatten()
    ops['ycup'], ops['xcup'] = ys, xs

    xc, yc = ops['xc'], ops['yc']
    Nfilt = len(ys)

    nC, nC2 = 10, 100
    iC, ds = nearest_chans(ys, yc, xs, xc, nC, device=device)
    iC2, ds2 = nearest_chans(ys, ys, xs, xs, nC2, device=device)

    ds_torch = torch.from_numpy(ds).to(device).float()
    weigh = torch.exp(- ds_torch.unsqueeze(-1) / (sig * (1+torch.arange(nsizes, device = device)))**2)
    weigh = torch.permute(weigh, (2, 0, 1)).contiguous()
    weigh = weigh / (weigh**2).sum(1).unsqueeze(1)**.5

    st = np.zeros((10**6, 6), 'float64')
    tF = np.zeros((10**6, nC , ops['nwaves']), 'float32')

    k = 0
    nt = ops['nt']
    tarange = torch.arange(-(nt//2),nt//2+1, device = device)
    s = StringIO()
    for ibatch in tqdm(np.arange(bfile.n_batches), miniters=200 if progress_bar else None, 
                        mininterval=60 if progress_bar else None):
        X = bfile.padded_batch_to_torch(ibatch, ops)

        xy, imax, amp, adist = template_match(X, ops, iC, iC2, weigh, device=device)
        yct = yweighted(yc, iC, adist, xy, device=device)
        nsp = len(xy)

        if k+nsp>st.shape[0]    :
            st = np.concatenate((st, np.zeros_like(st)), 0)
            tF = np.concatenate((tF, np.zeros_like(tF)), 0)

        xsub = X[iC[:,xy[:,:1]], xy[:,1:2] + tarange]
        xfeat = xsub @ ops['wPCA'].T
        tF[k:k+nsp] = xfeat.transpose(0,1).cpu().numpy()

        st[k:k+nsp,0] = ((xy[:,1]-nt)/ops['fs'] + ibatch * (ops['NT']/ops['fs'])).cpu().numpy()
        st[k:k+nsp,1] = yct.cpu().numpy()
        st[k:k+nsp,2] = amp.cpu().numpy()
        st[k:k+nsp,3] = imax.cpu().numpy()
        st[k:k+nsp,4] = ibatch
        st[k:k+nsp,5] = xy[:,0].cpu().numpy()

        k = k + nsp
        
        if progress_bar is not None:
            progress_bar.emit(int((ibatch+1) / bfile.n_batches * 100))
            
    st = st[:k]
    tF = tF[:k]
    ops['iC'] = iC
    ops['iC2'] = iC2
    ops['weigh'] = weigh
    return st, tF,ops
