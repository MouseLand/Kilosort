from torch.nn.functional import max_pool2d, avg_pool2d, conv1d, max_pool1d
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from kilosort.probe_utils import chan_weights


dev = torch.device('cuda')

def find_peaks(X, n_twav, init_threshold, loc_range = [5,5], long_range = [7,31]):
    Th = init_threshold
    Xneg = -X.unsqueeze(0)
    Xmax = max_pool2d(Xneg, (loc_range[0], 1), stride=1, padding=(loc_range[0]//2, 0))
    Xmax = max_pool2d(Xmax, (1, loc_range[1]), stride=1, padding=(0, loc_range[1]//2))

    peaks = torch.logical_and(Xmax == Xneg, Xmax > Th).float()
    peaks[0, :,:n_twav] = 0
    peaks[0, :,-n_twav:] = 0

    Pmax = avg_pool2d(peaks, (long_range[0], 1), stride=1, padding=(long_range[0]//2, 0))
    Pmax = avg_pool2d(Pmax, (1, long_range[1]), stride=1, padding=(0, long_range[1]//2))
    Pmax = Pmax * np.prod(long_range)

    is_peak = torch.logical_and(peaks[0], Pmax[0] < 1.2)
    xs = torch.nonzero(is_peak)
    return xs

def get_waves(bfile, init_threshold, n_wavpc, ops):
    dt = torch.arange(bfile.n_twav, device=dev) - bfile.twav_min
    clips = torch.zeros((100000, bfile.n_twav), device=dev)
    k = 0

    for j in range(bfile.n_batches):
        X = bfile.padded_batch_to_torch(j, ops)
        #X = preprocessing.load_transform(ops['filename'], j, ops, fwav = ops['fwav'], Wrot = ops['Wrot'])
        xs = find_peaks(X, bfile.n_twav, init_threshold)
        nsp = len(xs)
        if k+nsp>100000:
            break;
        clips[k:k+nsp] = X[xs[:,:1], xs[:,1:2] + dt]
        k += nsp

    clips = clips[:k]
    clips = clips/(clips**2).sum(1).unsqueeze(1)**.5
    clips = clips.cpu().numpy()
    
    model = TruncatedSVD(n_components = n_wavpc).fit(clips)
    wPCA = torch.from_numpy(model.components_).to(dev).float()
    
    model = KMeans(n_clusters = n_wavpc, n_init = 1).fit(clips)
    wTEMP = torch.from_numpy(model.cluster_centers_).to(dev).float()
    wTEMP = wTEMP / (wTEMP**2).sum(1).unsqueeze(1)**.5

    return wPCA, wTEMP

def template_match(X, wTEMP, n_twav, n_wavpc, spike_threshold, iC, iC2, weigh):
    NT = X.shape[-1]
    Nfilt = iC.shape[1]

    tch0 = torch.zeros(1,device = dev)
    tch1 = torch.ones(1,device = dev)

    W = wTEMP.unsqueeze(1)
    B = conv1d(X.unsqueeze(1), W, padding=n_twav//2)

    nt0 = 20
    nk = n_wavpc

    niter = 40
    nb = (NT-1)//niter+1
    As    = torch.zeros((Nfilt, NT), device=dev)
    Amaxs = torch.zeros((Nfilt, NT), device=dev)
    imaxs = torch.zeros((Nfilt, NT), dtype = torch.int64, device=dev)

    for t in range(niter):
        A = torch.einsum('ijk, jklm-> iklm', weigh, B[iC,:, nb*t:nb*(t+1)])
        A = A.transpose(0,1)
        A = A.reshape(Nfilt, -1, A.shape[-1])
        A, imax = torch.max(A, 1)
        As[:, nb*t:nb*(t+1)] = A
        imaxs[:, nb*t:nb*(t+1)] = imax
        Amax = torch.max(A[iC2], 0)[0]
        Amaxs[:, nb*t:nb*(t+1)] = Amax

    Amaxs[:,:n_twav] = 0
    Amaxs[:,-n_twav:] = 0
    Amaxs  = max_pool1d(Amaxs.unsqueeze(0), (2*nt0+1), stride = 1, padding = nt0).squeeze(0)
    xy = torch.logical_and(Amaxs==As, As > spike_threshold).nonzero()
    imax = imaxs[xy[:,0], xy[:,1]]
    amp = As[xy[:,0], xy[:,1]]
    adist = B[iC[:, xy[:,0]], imax%nk, xy[:,1]]

    return xy, imax, amp, adist

def yweighted(y_chan, iC, adist, xy):
    yy = torch.from_numpy(y_chan).to(dev)[iC]
    cF0 = torch.nn.functional.relu(adist)
    cF0 = cF0/cF0.sum(0)

    yct = (cF0 * yy[:,xy[:,0]]).sum(0)
    return yct

def template_centers(x_chan, y_chan):
    xmin, xmax, ymin, ymax = x_chan.min(), x_chan.max(), y_chan.min(), y_chan.max()
    dmin = np.median(np.diff(np.unique(y_chan)))
    yup = np.arange(ymin, ymax+.00001, dmin//2)

    yunq = np.unique(y_chan)
    mxc = np.NaN * np.ones(len(yunq))
    for j in range(len(yunq)):
        xc = x_chan[y_chan==yunq[j]]
        if len(xc)>1:
            mxc[j] = np.median(np.diff(np.sort(xc)))
    dminx = np.nanmedian(mxc)
    nx = np.round((xmax - xmin) / (dminx/2)) + 1
    xup = np.linspace(xmin, xmax, int(nx))
    return yup, xup, dmin, dminx

def nearest_chans(ys, y_chan, xs, x_chan, nC):
    ds = (ys - y_chan[:,np.newaxis])**2 + (xs - x_chan[:,np.newaxis])**2
    iC = np.argsort(ds, 0)[:nC]
    iC = torch.from_numpy(iC).to(dev)
    ds = np.sort(ds, 0)[:nC]
    return iC, ds

def chan_weights(x_chan, y_chan, nC=10, nC2=100):
    sig = 10
    nsizes = 5
    
    yup, xup, dmin, dminx = template_centers(x_chan, y_chan)
    [ys, xs] = np.meshgrid(yup, xup)
    ycup, xcup = ys.flatten(), xs.flatten()
    iC, ds = nearest_chans(ycup, y_chan, xcup, x_chan, nC)
    iC2, ds2 = nearest_chans(ycup, ycup, xcup, xcup, nC2)
    ds_torch = torch.from_numpy(ds).to(dev).float()
    weigh = torch.exp(- ds_torch.unsqueeze(-1) / (sig * (1+torch.arange(nsizes, device = dev)))**2)
    weigh = torch.permute(weigh, (2, 0, 1)).contiguous()
    weigh = weigh / (weigh**2).sum(1).unsqueeze(1)**.5
    return xcup, ycup, dmin, dminx, iC, iC2, weigh

def run(bfile, x_chan, y_chan, init_threshold, n_wavpc, spike_threshold, ops, dshift = None):        
    n_twav = bfile.n_twav

    wPCA, wTEMP = get_waves(bfile, init_threshold, n_wavpc, ops)
    
    nC=10
    xcup, ycup, dmin, dminx, iC, iC2, weigh = chan_weights(x_chan, y_chan, nC=nC, nC2=100)
    
    st = np.zeros((10**6, 6), 'float64')
    tF = np.zeros((10**6, nC , n_wavpc), 'float32')

    k = 0
    tarange = torch.arange(-(n_twav//2),n_twav//2+1, device = dev)

    for j in range(bfile.n_batches):
        X = bfile.padded_batch_to_torch(j, ops)
        #X = preprocessing.load_transform(ops['filename'], j, ops, fwav = ops['fwav'], 
        #                    Wrot = ops['Wrot'], dshift = dshift)
        xy, imax, amp, adist = template_match(X, wTEMP, n_twav, n_wavpc, spike_threshold, iC, iC2, weigh)
        yct = yweighted(y_chan, iC, adist, xy)
        nsp = len(xy)

        if k+nsp>st.shape[0]    :
            st = np.concatenate((st, np.zeros_like(st)), 0)
            tF = np.concatenate((tF, np.zeros_like(tF)), 0)

        xsub = X[iC[:,xy[:,:1]], xy[:,1:2] + tarange]
        xfeat = xsub @ wPCA.T
        tF[k:k+nsp] = xfeat.transpose(0,1).cpu().numpy()

        st[k:k+nsp,0] = ((xy[:,1]-n_twav) / bfile.fs + j * (bfile.batch_size / bfile.fs)).cpu().numpy()
        st[k:k+nsp,1] = yct.cpu().numpy()
        st[k:k+nsp,2] = amp.cpu().numpy()
        st[k:k+nsp,3] = imax.cpu().numpy()
        st[k:k+nsp,4] = j    
        st[k:k+nsp,5] = xy[:,0].cpu().numpy()

        k = k + nsp
        # time should go to seconds here. need amplitude and yct also. others less important. 

    st = st[:k]
    tF = tF[:k]
    
    ops['wPCA'] = wPCA
    ops['wTEMP'] = wTEMP
    ops['iC'] = iC 
    ops['iC2'] = iC2 
    ops['weigh'] = weigh 
    ops['xcup'] = xcup
    ops['ycup'] = ycup 
    ops['dmin'] = dmin 
    ops['dminx'] =dminx
    print(xcup)
    return st, tF, wPCA, wTEMP, iC, iC2, weigh, ops