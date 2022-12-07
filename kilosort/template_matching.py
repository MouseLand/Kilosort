import numpy as np
import spikedetect, preprocessing
import torch 
dev = torch.device('cuda')
from torch.nn.functional import conv1d, max_pool2d, max_pool1d


def Wupdate(X, U, stt, ops):
    nt = ops['nt']
    tiwave = torch.arange(-(nt//2), nt//2+1, device=dev) 
    xsub = X[:, stt[:,:1] + tiwave] @ ops['wPCA'].T
    Nfilt = U.shape[0]

    nsp = xsub.shape[1]
    M = torch.zeros((Nfilt, nsp), dtype=torch.float, device = dev)
    M[stt[:,1], torch.arange(nsp)] = 1

    Wsub = torch.einsum('ijk, lj -> lki', xsub, M)
    #Wsub = Wsub / (1e-3 + M.sum(1).unsqueeze(1))
    return Wsub, M.sum(1)

def prepare_extract(ops, U, nC):
    ds = (ops['xc'] - ops['xc'][:, np.newaxis])**2 +  (ops['yc'] - ops['yc'][:, np.newaxis])**2 
    iCC = np.argsort(ds, 0)[:nC]
    iCC = torch.from_numpy(iCC).to(dev)
    iU = torch.argmax((U**2).sum(1), -1)
    Ucc = U[torch.arange(U.shape[0]),:,iCC[:,iU]]
    return iCC, iU, Ucc


def extract(ops, U):
    nC = 10
    iCC, iU, Ucc = prepare_extract(ops, U, nC)
    ops['iCC'] = iCC
    ops['iU'] = iU

    nt = ops['nt']
    tiwave = torch.arange(-(nt//2), nt//2+1, device=dev) 

    ctc = prepare_matching(ops, U)
    st = np.zeros((10**6, 3), 'float64')
    tF  = torch.zeros((10**6, nC , ops['nwaves']))
    tF2 = torch.zeros((10**6, nC , ops['nwaves']))

    k = 0
    for ibatch in np.arange(ops['Nbatches']):
        X = preprocessing.load_transform(ops['filename'], ibatch, ops, fwav = ops['fwav'], 
                            Wrot = ops['Wrot'], dshift = ops['dshift'])
        
        #X0 = X.clone()

        stt, amps, Xres = run_matching(ops, X, U, ctc)

        xfeat2   = X[iCC[:, iU[stt[:,1:2]]],stt[:,:1] + tiwave] @ ops['wPCA'].T
        xfeat = Xres[iCC[:, iU[stt[:,1:2]]],stt[:,:1] + tiwave] @ ops['wPCA'].T
        xfeat += amps * Ucc[:,stt[:,1]]

        nsp = len(stt)   
        if k+nsp>st.shape[0]:                     
            st = np.concatenate((st, np.zeros_like(st)), 0)
            tF  = torch.cat((tF,  torch.zeros_like(tF)), 0)
            tF2 = torch.cat((tF2, torch.zeros_like(tF2)), 0)

        stt = stt.double()
        st[k:k+nsp,0] = ((stt[:,0]-nt)/ops['fs'] + ibatch * (ops['NT']/ops['fs'])).cpu().numpy()
        st[k:k+nsp,1] = stt[:,1].cpu().numpy()
        st[k:k+nsp,2] = amps[:,0].cpu().numpy()
        
        tF[k:k+nsp]  = xfeat.transpose(0,1).cpu()#.numpy()
        tF2[k:k+nsp] = xfeat2.transpose(0,1).cpu()

        k+= nsp
        if ibatch%100==0:
            print(ibatch)
    
    st = st[:k]
    tF = tF[:k]
    tF2 = tF2[:k]

    return st, tF, tF2, ops

def align_U(U, ops):
    Uex = torch.einsum('xyz, yt -> xtz', U, ops['wPCA'])
    X = Uex.reshape(-1, ops['Nchan']).T
    X = conv1d(X.unsqueeze(1), ops['wTEMP'].unsqueeze(1), padding=ops['nt']//2)
    Xmax = X.max(0)[0].max(0)[0].reshape(-1, ops['nt'])
    imax = torch.argmax(Xmax, 1)

    Unew = Uex.clone() 
    for j in range(ops['nt']):
        ix = imax==j
        Unew[ix] = torch.roll(Unew[ix], ops['nt']//2 - j, -2)
    Unew = torch.einsum('xty, zt -> xzy', Unew, ops['wPCA'])
    return Unew, imax

def run(ops):
    np.random.seed(101)
    iperm = np.random.permutation(ops['Nbatches'])
    niter = 40
    bb = ops['Nbatches']//niter
    Nchan = ops['Nchan']
    nPC = ops['wPCA'].shape[0]

    U = torch.zeros((0, nPC, Nchan), device = dev)

    for iiter in range(niter):
        isub = iperm[iiter*bb:(iiter+1)*bb]
        st, tF, Wres, nn, vexp = get_batch(ops, isub, U)    
        if U.shape[0]>0:
            U = U + Wres / (1e-6 + nn.unsqueeze(-1).unsqueeze(-1))
            U = U[nn>0]

        if iiter>niter-10:
            U, _ = align_U(U, ops)
            print(len(U), nn.sum().item(), len(st), vexp.item())
            continue

        Unew = get_new_templates(ops, tF, st)
        U = torch.cat((U, Unew), 0)
        print(len(U), len(Unew), nn.sum().item(), len(st), vexp.item())        
        
        U, _ = align_U(U, ops)
        
    return U


def get_batch(ops, isub, U):
    
    iC = ops['iC']
    #iC2 = ops['iC2']
    #weigh = ops['weigh']
    nt = ops['nt']
    Nchan = ops['Nchan']
    nPC = ops['wPCA'].shape[0]

    tarange = torch.arange(-(nt//2),nt//2+1, device = dev)

    nC = ops['iC'].shape[0]
    
    Nfilt = U.shape[0]
    if Nfilt>0:
        ctc = prepare_matching(ops, U)
    Wres = torch.zeros((Nfilt, nPC, Nchan), device = dev)
    nn = torch.zeros((Nfilt,), device = dev)

    tF = torch.zeros((10**5, nC , ops['nwaves']), device=dev)
    st = torch.zeros((10**5, 6), device=dev)
    k = 0
    vexp = 0
    tau = 400

    for ibatch in isub:
        #ibatch = iperm[ik*bb + j]
        X = preprocessing.load_transform(ops['filename'], ibatch, ops, fwav = ops['fwav'], 
                            Wrot = ops['Wrot'], dshift = ops['dshift'])
        
        if Nfilt>0:
            # do the template matching with W
            stt, B, Xres = run_matching(ops, X, U, ctc)

            # get updates for W
            Wres0, nn0 = Wupdate(Xres, U, stt, ops)
            
            nn   += nn0 #res[1]
            Wres += Wres0   

            #nn0 = nn0.unsqueeze(-1).unsqueeze(-1)
            #U = U + (1 - torch.exp(-nn0/tau)) * (Wres0/(1e-6 + nn0))
            #ctc = prepare_matching(ops, U)
        else:
            Xres = X


        # do the template matching on residuals
        xy, imax, amp, adist = spikedetect.template_match(Xres, ops, ops['iC'], ops['iC2'], ops['weigh'])
        
        xsub = Xres[iC[:,xy[:,:1]], xy[:,1:2] + tarange]
        xfeat = xsub @ ops['wPCA'].T

        nsp = len(xy)
        if k+nsp>tF.shape[0]:                     
            tF = torch.cat((tF, torch.zeros_like(tF)), 0)
            st = torch.cat((st, torch.zeros_like(st)), 0)

        tF[k:k+nsp] = xfeat.transpose(0,1)# .reshape(-1, nPC * Nchan)
        #st[k:k+nsp,0] = ((xy[:,1]-nt)/ops['fs'] + j * (ops['NT']/ops['fs']))
        #st[k:k+nsp,1] = yct
        st[k:k+nsp,2] = amp
        st[k:k+nsp,3] = imax
        st[k:k+nsp,4] = ibatch    
        st[k:k+nsp,5] = xy[:,0]

        k+= nsp

        vexp += (Xres**2).mean() / (X**2).mean()
    vexp = vexp/len(isub)

    tF = tF[:k]
    st = st[:k]

    return st, tF, Wres, nn, vexp

def get_new_templates(ops, tF, st):
    xcup, ycup = ops['xcup'], ops['ycup']
    Nchan = ops['Nchan']

    Th = ops['spkTh']
    kk = 0

    d0 = ops['dmin']
    ycent = np.arange(ycup.min()+d0-1, ycup.max()+d0+1, 2*d0)

    Wnew = torch.zeros((len(ycent), Nchan, 6), device = dev)

    igood = torch.ones((len(ycent)), dtype = torch.bool, device = dev)
    for kk in range(len(ycent)):
        # get the data
        Xd, ch_min, ch_max = get_data(ops, st, tF, ycent[kk], xcup.mean(), dmin = d0, dminx = ops['dminx'])
        #print(ch_min, ch_max, len(Xd))
        if Xd is None:
            igood[kk] = 0
            continue

        # find new spikes
        vexp = 2 * Xd @ Xd.T - (Xd**2).sum(1)
        vsum = (vexp * (vexp>Th**2)).sum(0)
        vmax, imax = torch.max(vsum, 0)
        mu  = Xd[vexp[:, imax] > Th**2].mean(0)

        Wnew[kk , ch_min:ch_max]  = mu.reshape((ch_max-ch_min, 6))

    Wnew = Wnew[igood]
    U = Wnew.transpose(2,1).contiguous()

    U, _ = remove_duplicates(ops, U, ns = None, icl = None)

    return U



def get_data(ops, st, tF, ycenter,  xcenter, dmin = 20, dminx = 32, ncomps = 64):
    iC = ops['iC']
    PID = st[:,5].long()
    xcup, ycup = ops['xcup'], ops['ycup']
    xy = np.vstack((xcup, ycup))
    xy = torch.from_numpy(xy).to(dev)
    
    y0 = ycenter # xy[1].mean() - ycenter
    x0 = xcenter #xy[0].mean() - xcenter

    #print(dmin, dminx)
    ix = torch.logical_and(torch.abs(xy[1] - y0) < dmin, torch.abs(xy[0] - x0) < dminx)
    #print(ix.nonzero()[:,0])
    igood = ix[PID].nonzero()[:,0]

    if len(igood)==0:
        return None, None,  None

    pid = st[igood, 5].int()
    data = tF[igood]
    nspikes, nchanraw, nfeatures = data.shape
    ichan = torch.unique(iC[:, ix])
    ch_min = torch.min(ichan)
    ch_max = torch.max(ichan)+1
    nchan = ch_max - ch_min

    dd = torch.zeros((nspikes, nchan, nfeatures), device = dev)
    for j in ix.nonzero()[:,0]:
        ij = (pid==j).nonzero()[:,0]
        #print(ij.sum())
        dd[ij.unsqueeze(-1), iC[:,j]-ch_min] = data[ij]

    Xd = torch.reshape(dd, (nspikes, -1))
    return Xd, ch_min, ch_max

def postprocess_templates(Wall, ops):

    Wall2, _ = align_U(Wall, ops)
    Wall3, _= remove_duplicates(ops, Wall2)

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


def paired_compare(ops, U):
    ctc = prepare_matching(ops, U)

    cc = ctc.max(-1)[0]
    cdiag = (1e-6 + torch.diag(cc))**.5
    cc = cc / torch.outer(cdiag, cdiag)
    cc = cc - torch.diag(torch.diag(cc))

    mu = (U**2).sum(-1).sum(-1)**.5

    dmu = 2 * (mu - mu.unsqueeze(1)) / (mu + mu.unsqueeze(1))
    return torch.logical_and(cc > .9, dmu<.2)

import CCG
def remove_duplicates(ops, U0, ns = None, icl = None, st = None):
    if icl is not None:
        iclust = icl.astype('int64')
    else:
        iclust = None
    U = U0.clone()

    if ns is None:
        ns = np.ones(U.shape[0], )
    
    CR = 0
    while 1:
        ikeep = np.ones(U.shape[0], bool)
        imerge = paired_compare(ops, U)
        if imerge.sum()==0:
            break

        while 1:
            xy = imerge.nonzero()
            if len(xy)==0:
                break
            xy = xy.cpu().numpy().astype('int64')
            x = xy[0,0]
            y = xy[0,1]

            ikeep[y] = 0

            imerge[x,:] = 0
            imerge[:,x] = 0
            imerge[y,:] = 0
            imerge[:,y] = 0

            U[x] = (U[x] * ns[x] + U[y] * ns[y]) / (ns[x] + ns[y])
            ns[x] = ns[x] + ns[y]
            if icl is not None:                
                st1 = st[iclust==y]
                st2 = st[iclust==x]
                cross_refractory = CCG.check_CCG(st1,  st2, nbins = 500, tbin  = 1/1000)[1]
                CR += cross_refractory
                iclust[iclust==y] = x

        #import pdb; pdb.set_trace()        
        U = U[ikeep]
        ns = ns[ikeep]
        if icl is not None:
            isum = np.cumsum(ikeep) - 1
            iclust  = isum[iclust]

    if icl is not None:
        print(CR)
    return U, iclust





def run_matching(ops, X, U, ctc):
    Th = ops['spkTh']
    nt = ops['nt']
    W = ops['wPCA'].contiguous()

    nm = (U**2).sum(-1).sum(-1)
    #mu = nm**.5 
    #U2 = U / mu.unsqueeze(-1).unsqueeze(-1)

    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)
    B = torch.einsum('ijk, kjl -> il', U, B)

    trange = torch.arange(-nt, nt+1, device=dev) 
    tiwave = torch.arange(-(nt//2), nt//2+1, device=dev) 

    st = torch.zeros((100000,2), dtype = torch.int64, device = dev)
    amps = torch.zeros((100000,1), dtype = torch.float, device = dev)
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