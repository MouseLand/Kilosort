from scipy.sparse import csr_matrix
import numpy as np


def cluster_qr(M, iclust, iclust0):
    NN = M.shape[0]
    nr = M.shape[1]

    nc = iclust.max()+1
    q = csr_matrix((np.ones(NN,), (iclust, np.arange(NN))), (nc, NN))
    r  = csr_matrix((np.ones(nr,), (np.arange(nr), iclust0)), (nr, nc))
    return q,r

def Mstats(M):
    m = M.sum()
    ki = np.array(M.sum(1)).flatten()
    kj = np.array(M.sum(0)).flatten()
    ki = m * ki/ki.sum()
    kj = m * kj/kj.sum()
    return m, ki, kj

def prepare(M, iclust, iclust0, lam=1):
    m, ki, kj = Mstats(M)
    q,r = cluster_qr(M, iclust, iclust0)
    cc = (q @ M @ r).toarray()
    nc = cc.shape[0]
    cneg = .001 + np.outer(q @ ki , kj @ r)/m
    return cc, cneg

def merge_reduce(cc, cneg, iclust):
    nmerges = 0
    nc = cc.shape[0]

    cc = cc + cc.T
    cneg = cneg + cneg.T

    crat = cc/cneg #(cc + cc.T)/ (cneg + cneg.T)
    crat = crat -np.diag(np.diag(crat)) - np.eye(crat.shape[0])

    xtree, tstat = find_merges(crat, cc, cneg)

    my_clus = get_my_clus(xtree, tstat)
    return xtree, tstat, my_clus

def find_merges(crat, cc, cneg):
    nc = cc.shape[0]
    xtree = np.zeros((nc-1,3), 'int32')
    tstat = np.zeros((nc-1,3), 'float32')
    xnow = np.arange(nc)
    ntot = np.ones(nc,)

    for nmerges in range(nc-1):
        y, x = np.unravel_index(np.argmax(crat), cc.shape)
        lam = crat[y,x]

        m      = cc[y,x] + cc[x,x] + cc[x,y] + cc[y,x]
        ki = cc[x,x] + cc[x,y]
        kj = cc[y,y] + cc[y,x]
        cneg_l = .5 * (ki * kj + (m-ki) * (m-kj)) / m
        cpos_l = cc[y,x] + cc[x,y]
        M      = cpos_l / cneg_l

        cc[y]   = cc[y] + cc[x]
        cc[:,y] = cc[:,y] + cc[:,x]
        cc[x]   = -1
        cc[:,x] = -1
        cneg[y]   = cneg[y]   + cneg[x]
        cneg[:,y] = cneg[:,y] + cneg[:,x]

        crat[y] = cc[y]/cneg[y]
        crat[:,y] = crat[y]
        crat[y,y] = -1
        crat[x] = -1
        crat[:,x]=-1

        xtree[nmerges,:] = [xnow[x], xnow[y], nmerges + nc]
        tstat[nmerges,:] = [lam, ntot[x]+ntot[y], M]

        ntot[y] +=ntot[x]
        xnow[y] = nc+nmerges

    return xtree, tstat

def get_my_clus(xtree, tstat):
    nc = xtree.shape[0]+1
    my_clus = [[j] for j in range(nc)]
    for t in range(nc-1):
        new_clus = my_clus[xtree[t,1]].copy()
        new_clus.extend(my_clus[xtree[t,0]])
        my_clus.append(new_clus)
    return my_clus

def maketree(M, iclust, iclust0):

    #m, ki, kj = Mstats(M)
    #iclust = swarmer.assign_iclust(M, ki, kj, m, iclust[::nskip], lam = 1)
    #iclust, nc  = swarmer.cleanup_index(iclust)

    nc = np.max(iclust) + 1

    cc, cneg        = prepare(M, iclust, iclust0, lam = 1)
    xtree, tstat, my_clus  = merge_reduce(cc, cneg, iclust)

    return xtree, tstat, my_clus
