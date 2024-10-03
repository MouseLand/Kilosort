import logging

import numpy as np
import torch 
from torch.nn.functional import conv1d, max_pool2d, max_pool1d
from tqdm import tqdm

from kilosort import CCG
from kilosort.utils import log_performance

logger = logging.getLogger(__name__)


def prepare_extract(ops, U, nC, device=torch.device('cuda')):
    ds = (ops['xc'] - ops['xc'][:, np.newaxis])**2 +  (ops['yc'] - ops['yc'][:, np.newaxis])**2 
    iCC = np.argsort(ds, 0)[:nC]
    iCC = torch.from_numpy(iCC).to(device)
    iU = torch.argmax((U**2).sum(1), -1)
    Ucc = U[torch.arange(U.shape[0]),:,iCC[:,iU]]
    return iCC, iU, Ucc

def extract(ops, bfile, U, device=torch.device('cuda'), progress_bar=None):
    nC = ops['settings']['nearest_chans']
    iCC, iU, Ucc = prepare_extract(ops, U, nC, device=device)
    ops['iCC'] = iCC
    ops['iU'] = iU
    nt = ops['nt']
    
    tiwave = torch.arange(-(nt//2), nt//2+1, device=device) 
    ctc = prepare_matching(ops, U)
    st = np.zeros((10**6, 3), 'float64')
    tF  = torch.zeros((10**6, nC , ops['settings']['n_pcs']))
    k = 0
    for ibatch in tqdm(np.arange(bfile.n_batches), miniters=200 if progress_bar else None, 
                        mininterval=60 if progress_bar else None):
        if ibatch % 100 == 0:
            log_performance(logger, 'debug', f'Batch {ibatch}')

        X = bfile.padded_batch_to_torch(ibatch, ops)
        stt, amps, Xres = run_matching(ops, X, U, ctc, device=device)
        xfeat = Xres[iCC[:, iU[stt[:,1:2]]],stt[:,:1] + tiwave] @ ops['wPCA'].T
        xfeat += amps * Ucc[:,stt[:,1]]

        if ibatch == 0:
            # Can sometimes get negative spike times for first batch since
            # we're aligning to nt0min, not nt//2, but these should be discarded.
            neg_spikes = (stt[:,0] - nt - nt//2 + ops['nt0min']) < 0
            stt = stt[~neg_spikes,:]
            xfeat = xfeat[:,~neg_spikes,:]
            amps = amps[~neg_spikes,:]

        nsp = len(stt) 
        if k+nsp>st.shape[0]:                     
            st = np.concatenate((st, np.zeros_like(st)), 0)
            tF  = torch.cat((tF,  torch.zeros_like(tF)), 0)

        stt = stt.double()
        st[k:k+nsp,0] = ((stt[:,0]-nt) + ibatch * (ops['batch_size'])).cpu().numpy() - nt//2 + ops['nt0min']
        st[k:k+nsp,1] = stt[:,1].cpu().numpy()
        st[k:k+nsp,2] = amps[:,0].cpu().numpy()
        
        tF[k:k+nsp]  = xfeat.transpose(0,1).cpu()

        k+= nsp
        
        if progress_bar is not None:
            progress_bar.emit(int((ibatch+1) / bfile.n_batches * 100))

    log_performance(logger, 'debug', f'Batch {ibatch}')

    isort = np.argsort(st[:k,0])
    st = st[isort]
    tF = tF[isort]

    return st, tF, ops

def align_U(U, ops, device=torch.device('cuda')):
    Uex = torch.einsum('xyz, zt -> xty', U.to(device), ops['wPCA'])
    X = Uex.reshape(-1, ops['Nchan']).T
    X = conv1d(X.unsqueeze(1), ops['wTEMP'].unsqueeze(1), padding=ops['nt']//2)
    Xmax = X.abs().max(0)[0].max(0)[0].reshape(-1, ops['nt'])
    imax = torch.argmax(Xmax, 1)

    Unew = Uex.clone() 
    for j in range(ops['nt']):
        ix = imax==j
        Unew[ix] = torch.roll(Unew[ix], ops['nt']//2 - j, -2)
    Unew = torch.einsum('xty, zt -> xzy', Unew, ops['wPCA'])#.transpose(1,2).cpu()
    return Unew, imax


def postprocess_templates(Wall, ops, clu, st, device=torch.device('cuda')):
    Wall2, _ = align_U(Wall, ops, device=device)
    #Wall3, _= remove_duplicates(ops, Wall2)
    Wall3, _, _ = merging_function(ops, Wall2.transpose(1,2), clu, st[:,0],
                                   0.9, 'mu', device=device)
    Wall3 = Wall3.transpose(1,2).to(device)
    return Wall3

def prepare_matching(ops, U):
    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])

    #mu = (U**2).sum(-1).sum(-1)**.5
    #U2 = U / mu.unsqueeze(-1).unsqueeze(-1)

    UtU = torch.einsum('ikl, jml -> ijkm',  U, U)
    ctc = torch.einsum('ijkm, kml -> ijl', UtU, WtW)

    return ctc

def run_matching(ops, X, U, ctc, device=torch.device('cuda')):
    Th = ops['Th_learned']
    nt = ops['nt']
    W = ops['wPCA'].contiguous()

    nm = (U**2).sum(-1).sum(-1)
    #mu = nm**.5 
    #U2 = U / mu.unsqueeze(-1).unsqueeze(-1)

    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)
    B = torch.einsum('ijk, kjl -> il', U, B)

    trange = torch.arange(-nt, nt+1, device=device) 
    tiwave = torch.arange(-(nt//2), nt//2+1, device=device) 

    st = torch.zeros((100000,2), dtype = torch.int64, device = device)
    amps = torch.zeros((100000,1), dtype = torch.float, device = device)
    k = 0

    Xres = X.clone()
    lam = 20

    for t in range(100):
        # Cf = 2 * B - nm.unsqueeze(-1) 
        Cf = torch.relu(B)**2 /nm.unsqueeze(-1)
        #a = 1 + lam
        #b = torch.relu(B) + lam * mu.unsqueeze(-1)
        #Cf = b**2 / a - lam * mu.unsqueeze(-1)**2

        Cf[:, :nt] = 0
        Cf[:, -nt:] = 0

        Cfmax, imax = torch.max(Cf, 0)
        Cmax  = max_pool1d(Cfmax.unsqueeze(0).unsqueeze(0), (2*nt+1), stride = 1, padding = (nt))

        #print(Cfmax.shape)
        #import pdb; pdb.set_trace()
        cnd1 = Cmax[0,0] > Th**2
        cnd2 = torch.abs(Cmax[0,0] - Cfmax) < 1e-9
        xs = torch.nonzero(cnd1 * cnd2)

        if len(xs)==0:
            #print('iter %d'%t)
            break

        iX = xs[:,:1]
        iY = imax[iX]

        #isort = torch.sort(iX)

        nsp = len(iX)
        st[k:k+nsp, 0] = iX[:,0]
        st[k:k+nsp, 1] = iY[:,0]
        amps[k:k+nsp] = B[iY,iX] / nm[iY]
        amp = amps[k:k+nsp]

        k+= nsp

        #amp = B[iY,iX] 

        n = 2
        for j in range(n):
            Xres[:, iX[j::n] + tiwave]  -= amp[j::n] * torch.einsum('ijk, jl -> kil', U[iY[j::n,0]], W)
            B[   :, iX[j::n] + trange]  -= amp[j::n] * ctc[:,iY[j::n,0],:]

    st = st[:k]
    amps = amps[:k]

    return  st, amps, Xres


def merging_function(ops, Wall, clu, st, r_thresh=0.5, mode='ccg', device=torch.device('cuda')):
    clu2 = clu.copy()
    clu_unq, ns = np.unique(clu2, return_counts = True)

    Ww = Wall.to(device)
    NN = len(Ww)

    isort = np.argsort(ns)[::-1]

    is_merged = np.zeros(NN, 'bool')
    is_good = np.zeros(NN,)

    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    if mode == 'ccg':
        is_ref, est_contam_rate = CCG.refract(clu, st/ops['fs'],
                                              acg_threshold=acg_threshold,
                                              ccg_threshold=ccg_threshold)

    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])

    t = 0
    nmerge = 0
    while t<NN:
        #if t%100==0:
            #print(t, nmerge)

        kk = clu_unq[isort[t]]

        if (mode == 'ccg') and is_ref[kk]==0:
            t += 1
            continue

        if is_merged[kk]:            
            t += 1
            continue

        mu = (Ww**2).sum((1,2), keepdims=True)**.5
        Wnorm = Ww / (1e-6 + mu)

        UtU = torch.einsum('lk, jlm -> jkm',  Wnorm[kk], Wnorm)
        ctc = torch.einsum('jkm, kml -> jl', UtU, WtW)

        cmax = ctc.max(1)[0]
        cmax[kk] = 0

        jsort = np.argsort(cmax.cpu().numpy())[::-1]

        if mode == 'ccg':
            st0 = st[clu2==kk] / ops['fs']
        
        is_ccg  = 0
        for j in range(NN):
            jj = jsort[j]
            if cmax[jj] < r_thresh:
                break
            # compare with CCG
            if mode == 'ccg':
                st1 = st[clu2==jj] / ops['fs']
                _, is_ccg, _ = CCG.check_CCG(st0, st1, acg_threshold=acg_threshold,
                                             ccg_threshold=ccg_threshold)        
            else:
                dmu = 2 * (mu[kk] - mu[jj]) / (mu[kk] + mu[jj])
                is_ccg = dmu.abs() < 0.2

            if is_ccg:
                is_merged[jj] = 1
                Ww[kk] = ns[kk]/(ns[kk]+ns[jj]) * Ww[kk] + ns[jj]/(ns[kk]+ns[jj]) * Ww[jj]            
                Ww[jj] = 0

                ns[kk] += ns[jj]
                ns[jj] = 0
                clu2[clu2==jj] = kk            

                break

        if is_ccg==0:            
            t +=1    
        else:                
            nmerge+=1
    
    imap = np.cumsum((~is_merged).astype('int32')) - 1
    if imap.size > 0:
        # Otherwise, everything has been merged into a single cluster
        clu2 = imap[clu2]

    Ww = Ww[~is_merged]

    if mode == 'ccg':
        is_ref = is_ref[~is_merged]
    else:
        is_ref = None

    return Ww.cpu(), clu2, is_ref
