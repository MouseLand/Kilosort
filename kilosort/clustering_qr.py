import numpy as np
import torch
from torch import sparse_coo_tensor as coo
dev = torch.device('cuda')
from scipy.sparse import csr_matrix 

import faiss
import hierarchical, swarmsplitter 

def neigh_mat(Xd, nskip = 10, n_neigh=30):
    Xsub = Xd[::nskip]
    n_samples, dim = Xd.shape
    n_nodes = Xsub.shape[0]

    Xd = np.ascontiguousarray(Xd)
    Xsub = np.ascontiguousarray(Xsub)
    index = faiss.IndexFlatL2(dim)   # build the index
    index.add(Xsub)    # add vectors to the index
    dn, kn = index.search(Xd, n_neigh)     # actual search

    #kn = kn[:,1:]
    n_neigh = kn.shape[-1]
    dexp = np.ones(kn.shape, np.float32)
    #M   = csr_matrix((dexp.flatten(), kn.flatten(), n_neigh*np.arange(kn.shape[0]+1)),
     #              (kn.shape[0], Xsub.shape[0]))

    rows = np.tile(np.arange(n_samples)[:, np.newaxis], (1, n_neigh)).flatten()
    M   = csr_matrix((dexp.flatten(), (rows, kn.flatten())),
                   (kn.shape[0], n_nodes))

    M[np.arange(0,n_samples,nskip), np.arange(n_nodes)] = 0

    return kn, M


def assign_mu(iclust, Xg, cols_mu, tones, nclust = None, lpow = 1):
    NN, nfeat = Xg.shape

    rows = iclust.unsqueeze(-1).tile((1,nfeat))
    ii = torch.vstack((rows.flatten(), cols_mu.flatten()))
    iin = torch.vstack((rows[:,0], cols_mu[:,0]))
    if lpow==1:
        C = coo(ii, Xg.flatten(), (nclust, nfeat))
    else:
        C = coo(ii, (Xg**lpow).flatten(), (nclust, nfeat))
    N = coo(iin, tones, (nclust, 1))
    C = C.to_dense()
    N = N.to_dense()
    mu = C / (1e-6 + N)

    return mu, N

def assign_iclust_slow(Xg, mu, rows_neigh, isub, kn, tones2, lam = 1):
    NN, nfeat = Xg.shape
    nclust = mu.shape[0]
    n_neigh = kn.shape[-1]

    ij = torch.vstack((rows_neigh.flatten(), isub[kn].flatten()))
    xN = coo(ij, (1e-12 + tones2).flatten(), (NN, nclust))
    xN = xN.coalesce()
    
    values  = xN.values().cpu().numpy()
    indices = xN.indices().cpu().numpy()
    from scipy.sparse import coo_matrix
    C = coo_matrix((values, indices), (NN, nclust)).tocsr()
    vexp = C[np.arange(NN)[:, np.newaxis], isub[kn].cpu().numpy()].todense()
    vexp = torch.from_numpy(vexp).to(dev)

    # too much memory, this can be a loop
    vv = torch.einsum('ij, ikj -> ik', Xg , mu[isub[kn]])
    nm = (mu**2).sum(1)[isub[kn]]
    vexp = vexp + 2*vv-nm
    
    tarange = torch.arange(NN, device = dev)#.tile(1,n_neigh)    
    imax = torch.argmax(vexp, 1)
    iclust = isub[kn[tarange, imax]]

    return iclust

def assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj):
    NN = kn.shape[0]

    ij = torch.vstack((rows_neigh.flatten(), isub[kn].flatten()))
    xN = coo(ij, tones2.flatten(), (NN, nclust))
    xN = xN.to_dense()

    if lam > 0:
        tones = torch.ones(len(kj), device = dev)
        tzeros = torch.zeros(len(kj), device = dev)
        ij = torch.vstack((tzeros, isub))    
        kN = coo(ij, tones, (1, nclust))
    
        xN = xN - lam/m * (ki.unsqueeze(-1) * kN.to_dense()) 
    
    iclust = torch.argmax(xN, 1)

    return iclust

def assign_isub(iclust, kn, tones2, nclust, nsub, lam, m,ki,kj):
    n_neigh = kn.shape[1]
    cols = iclust.unsqueeze(-1).tile((1, n_neigh))
    iis = torch.vstack((kn.flatten(), cols.flatten()))

    xS = coo(iis, tones2.flatten(), (nsub, nclust))
    xS = xS.to_dense()

    if lam > 0:
        tones = torch.ones(len(ki), device = dev)
        tzeros = torch.zeros(len(ki), device = dev)
        ij = torch.vstack((tzeros, iclust))    
        kN = coo(ij, tones, (1, nclust))
        xS = xS - lam / m * (kj.unsqueeze(-1) * kN.to_dense())

    isub = torch.argmax(xS, 1)
    return isub

def Mstats(M):
    m = M.sum()
    ki = np.array(M.sum(1)).flatten()
    kj = np.array(M.sum(0)).flatten()
    ki = m * ki/ki.sum()
    kj = m * kj/kj.sum()

    ki = torch.from_numpy(ki).to(dev)
    kj = torch.from_numpy(kj).to(dev)
    
    return m, ki, kj


def cluster(Xd, iclust = None, kn = None, nskip = 20, n_neigh = 10, nclust = 200, seed = 1, niter = 200, lam = 0):    

    if kn is None:
        kn, M = neigh_mat(Xd, nskip = nskip, n_neigh = n_neigh)

    m, ki, kj = Mstats(M)

    #Xg = torch.from_numpy(Xd).to(dev)
    Xg = Xd.to(dev)
    kn = torch.from_numpy(kn).to(dev)

    n_neigh = kn.shape[1]
    NN, nfeat = Xg.shape
    nsub = (NN-1)//nskip + 1

    rows_neigh = torch.arange(NN, device = dev).unsqueeze(-1).tile((1,n_neigh))
    
    tones2 = torch.ones((NN, n_neigh), device = dev)

    if iclust is None:
        iclust_init =  kmeans_plusplus(Xg, niter = nclust, seed = seed)
        iclust = iclust_init.clone()
    else:
        iclust_init = iclust.clone()
        
    for t in range(niter):
        # given iclust, reassign isub
        isub = assign_isub(iclust, kn, tones2, nclust , nsub, lam, m,ki,kj)

        # given mu and isub, reassign iclust
        iclust = assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj)
    
    _, iclust = torch.unique(iclust, return_inverse=True)    
    nclust = iclust.max() + 1
    isub = assign_isub(iclust, kn, tones2, nclust , nsub, lam, m,ki,kj)

    iclust = iclust.cpu().numpy()
    isub = isub.cpu().numpy()

    return iclust, isub, M, iclust_init


def kmeans_plusplus(Xg, niter = 200, seed = 1):
    #Xg = torch.from_numpy(Xd).to(dev)    
    vtot = (Xg**2).sum(1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    ntry = 100
    NN, nfeat = Xg.shape    
    mu = torch.zeros((niter, nfeat), device = dev)
    vexp0 = torch.zeros(NN,device = dev)

    iclust = torch.zeros((NN,), dtype = torch.int, device = dev)
    for j in range(niter):
        v2 = torch.relu(vtot - vexp0) 
        isamp = torch.multinomial(v2, ntry)
        
        Xc = Xg[isamp]    
        vexp = 2 * Xg @ Xc.T - (Xc**2).sum(1)
        
        dexp = vexp - vexp0.unsqueeze(1)
        dexp = torch.relu(dexp)
        vsum = dexp.sum(0)

        imax = torch.argmax(vsum)
        ix = dexp[:, imax] > 0 

        mu[j] = Xg[ix].mean(0)
        vexp0[ix] = vexp[ix,imax]
        iclust[ix] = j

    return iclust

def collapse_neighbors(iclust, kn, tones2, isub, nclust = None):
    n_neigh = kn.shape[1]
    cols = iclust.unsqueeze(-1).tile((1, n_neigh))
    rows = isub[kn]
    iis = torch.vstack((cols.flatten(), rows.flatten()))

    ccN = coo(iis, tones2.flatten(), (nclust, nclust))
    ccN = ccN.to_dense()
    return ccN


def big_merge(Xg, cols_mu, iclust, kn, tones, tones2, nsub, lam):
    _, iclust = torch.unique(iclust, return_inverse=True)
    nclust = iclust.max() + 1
    isub = assign_isub(iclust, kn, tones2, nclust = nclust, nsub=nsub)

    ccN = collapse_neighbors(iclust, kn, tones2, isub, nclust = nclust)
    #ccN = ccN + ccN.T
    ccN = ccN - torch.diag(torch.diag(ccN))

    mu, N  = assign_mu(iclust, Xg, cols_mu, tones, nclust = nclust)
    mu2, N = assign_mu(iclust, Xg, cols_mu, tones, nclust = nclust, lpow = 2)
    nsum = N.sum()
    N = N/nsum
    ccN = ccN/nsum

    while 1:
        score = compute_score(mu,mu2,N,ccN, lam)
        if score.max() <.00001:
            break
        ki,kj = np.unravel_index(score.argmax().cpu(), score.shape)
        
        # update mu
        mu[ki]  = (mu[ki]  * N[ki] +  mu[kj] * N[kj]) / (N[ki] + N[kj])
        mu2[ki] = (mu2[ki] * N[ki] + mu2[kj] * N[kj]) / (N[ki] + N[kj])
        mu[kj] = 0
        mu2[kj] = 0

        # update N
        N[ki] = N[ki] + N[kj]
        N[kj] = 0

        # update ccN
        ccN[ki]   = ccN[ki]   + ccN[kj]
        ccN[:,ki] = ccN[:,ki] + ccN[:,kj]
        ccN[ki,ki] = 0
        ccN[kj] = 0
        ccN[:,kj] = 0

        iclust[iclust==kj] = ki
    return iclust 


def compute_score(mu, mu2, N, ccN, lam):
    mu_pairs  = ((N*mu).unsqueeze(1)  + N*mu)  / (1e-6 + N+N[:,0]).unsqueeze(-1)
    mu2_pairs = ((N*mu2).unsqueeze(1) + N*mu2) / (1e-6 + N+N[:,0]).unsqueeze(-1)

    vpair = (mu2_pairs - mu_pairs**2).sum(-1) * (N + N[:,0])
    vsingle = N[:,0] * (mu2 - mu**2).sum(-1)
    dexp = vpair - (vsingle + vsingle.unsqueeze(-1))

    dexp = dexp - torch.diag(torch.diag(dexp))

    score = (ccN + ccN.T) - lam * dexp
    return score

def run_one(Xd, st0, nskip = 20, lam = 0):

    iclust, iclust0, M = cluster(Xd,nskip = nskip, lam = 0, seed = 5)

    xtree, tstat, my_clus = hierarchical.maketree(M, iclust, iclust0)

    xtree, tstat = swarmsplitter.split(Xd.numpy(), xtree, tstat, iclust, my_clus, meta = st0)

    iclust1 = swarmsplitter.new_clusters(iclust, my_clus, xtree, tstat)

    return iclust1

import time
def run(ops, st, tF):
    xcup, ycup = ops['xcup'], ops['ycup']

    d0 = ops['dmin']
    ycent = np.arange(ycup.min()+d0-1, ycup.max()+d0+1, 2*d0)

    nsp = st.shape[0]
    clu = np.zeros(nsp, 'int32')
    nmax = 0

    nskip = 20

    Wall = torch.zeros((0, ops['Nchan'], ops['nwaves']))
    t0 = time.time()
    for kk in range(len(ycent)):
        # get the data
        iclust_template = st[:,1].astype('int32')

        Xd, ch_min, ch_max, igood  = get_data_cpu(ops, iclust_template, tF, ycent[kk],  
                                    xcup.mean(), dmin = d0, dminx = ops['dminx'], ncomps = 64)

        if Xd is None:            
            continue

        if Xd.shape[0]<1000:
            #clu[igood] = nmax
            #nmax += 1
            iclust = torch.zeros((Xd.shape[0],))
        else:
            st0 = st[igood,0]

            # find new clusters
            iclust, iclust0, M, iclust_init = cluster(Xd,nskip = nskip, lam = 1, seed = 5)

            xtree, tstat, my_clus = hierarchical.maketree(M, iclust, iclust0)

            xtree, tstat = swarmsplitter.split(Xd.numpy(), xtree, tstat, iclust, my_clus) #, meta = st0)

            iclust = swarmsplitter.new_clusters(iclust, my_clus, xtree, tstat)

        clu[igood] = iclust + nmax
        Nfilt = int(iclust.max() + 1)
        nmax += Nfilt

        # we need the new templates here         
        W = torch.zeros((Nfilt, ops['Nchan'], ops['nwaves']))
        for j in range(Nfilt):
            w = Xd[iclust==j].mean(0)
            W[j, ch_min:ch_max, :] = torch.reshape(w, (-1, ops['nwaves'])).cpu()
        
        Wall = torch.cat((Wall, W), 0)

        if kk%10==0:
            print(kk, nmax, time.time()-t0)

    return clu, Wall


def get_data_cpu(ops, PID, tF, ycenter,  xcenter, dmin = 20, dminx = 32, ncomps = 64):
    PID =  torch.from_numpy(PID).long()

    iU = ops['iU'].cpu().numpy()
    iC = ops['iCC'][:, ops['iU']]
    #PID = st[:,5].long()
    xcup, ycup = ops['xc'][iU], ops['yc'][iU]
    xy = np.vstack((xcup, ycup))
    xy = torch.from_numpy(xy)
    
    y0 = ycenter # xy[1].mean() - ycenter
    x0 = xcenter #xy[0].mean() - xcenter

    #print(dmin, dminx)
    ix = torch.logical_and(torch.abs(xy[1] - y0) < dmin, torch.abs(xy[0] - x0) < dminx)
    #print(ix.nonzero()[:,0])
    igood = ix[PID].nonzero()[:,0]

    if len(igood)==0:
        return None, None,  None, None

    pid = PID[igood]
    data = tF[igood]
    nspikes, nchanraw, nfeatures = data.shape
    ichan = torch.unique(iC[:, ix])
    ch_min = torch.min(ichan)
    ch_max = torch.max(ichan)+1
    nchan = ch_max - ch_min

    dd = torch.zeros((nspikes, nchan, nfeatures))
    for j in ix.nonzero()[:,0]:
        ij = torch.nonzero(pid==j)[:, 0]
        #print(ij.sum())
        dd[ij.unsqueeze(-1), iC[:,j]-ch_min] = data[ij]

    Xd = torch.reshape(dd, (nspikes, -1))
    return Xd, ch_min, ch_max, igood



def assign_clust(rows_neigh, iclust, kn, tones2, nclust):    
    NN = len(iclust)

    ij = torch.vstack((rows_neigh.flatten(), iclust[kn].flatten()))
    xN = coo(ij, tones2.flatten(), (NN, nclust))
    
    xN = xN.to_dense() 
    iclust = torch.argmax(xN, 1)

    return iclust

def assign_iclust0(Xg, mu):
    vv = Xg @ mu.T
    nm = (mu**2).sum(1)
    iclust = torch.argmax(2*vv-nm, 1)
    return iclust
