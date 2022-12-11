import numpy as np
from numba import njit
import math

def count_elements(kk, iclust, my_clus, xtree):
    n1 = np.isin(iclust, my_clus[xtree[kk, 0]]).sum()
    n2 = np.isin(iclust, my_clus[xtree[kk, 1]]).sum()
    return n1, n2

def check_split(Xd, kk, xtree, iclust, my_clus):
    ixy = np.isin(iclust, my_clus[xtree[kk, 2]])
    iclu = iclust[ixy]
    labels = 2*np.isin(iclu, my_clus[xtree[kk, 0]]) - 1

    Xs = Xd[ixy]
    Xs[:,-1] = 1

    w = np.ones((Xs.shape[0],1))
    w[labels>0] = np.mean(labels<0)
    w[labels<0] = np.mean(labels>0)

    CC = Xs.T @ (Xs * w)
    CC = CC + .01 * np.eye(CC.shape[0])
    b = np.linalg.solve(CC, labels @ (Xs * w))
    xproj = Xs @ b

    score = bimod_score(xproj)
    return xproj, score

def clean_tree(valid_merge, xtree, inode):
    ix = (xtree[:,2]==inode).nonzero()[0]
    if len(ix)==0:
        return
    valid_merge[ix] = 0
    clean_tree(valid_merge, xtree, xtree[ix, 0])
    clean_tree(valid_merge, xtree, xtree[ix, 1])
    return

def bimod_score(xproj):
    from scipy.ndimage import gaussian_filter1d
    xbin, _ = np.histogram(xproj, np.linspace(-2,2,400))
    xbin = gaussian_filter1d(xbin.astype('float32'), 4)

    imin = np.argmin(xbin[175:225])
    xmin = np.min(xbin[175:225])
    xm1  = np.max(xbin[:imin+175])
    xm2  = np.max(xbin[imin+175:])

    score = 1 - np.maximum(xmin/xm1, xmin/xm2)
    return score

@njit()
def compute_CCG(st1, st2, tbin = 1/1000, nbins = 500):

    st1 = np.sort(st1)
    st2 = np.sort(st2)

    dt = nbins * tbin
    T = np.maximum(st1.max(), st2.max()) - np.minimum(st1.min(), st2.min())

    ilow = 0
    ihigh = 0
    j = 0

    K = np.zeros(2*nbins+1,)
    while j<len(st2):
        while (ihigh<len(st1)) and (st1[ihigh] <= st2[j]+dt):
            ihigh += 1
        while (ilow<len(st1))  and (st1[ilow] <= st2[j]-dt):
            ilow += 1
        if ilow >=len(st1):
            break
        if st1[ilow]>st2[j]+dt:
            j += 1
            continue;
        for k in range(ilow, ihigh):
            ibin = int(np.round((st2[j] - st1[k])/tbin))
            K[ibin+nbins] += 1
        j += 1
    return K, T

#@njit()
def CCG_metrics(st1, st2, K, T, nbins=None, tbin=None):
    irange1 = np.hstack((np.arange(1,nbins//2), np.arange(3*nbins//2, 2*nbins)))
    irange2 = np.arange(nbins-50, nbins-10)
    irange3 = np.arange(nbins+10, nbins+50)

    Q00 = K[irange1].sum() / (len(irange1) * tbin * len(st1) * len(st2) /T)
    Q1 = K[irange2].sum() / (len(irange2) * tbin * len(st1) * len(st2) /T)
    Q2 = K[irange3].sum() / (len(irange3) * tbin * len(st1) * len(st2) /T)
    Q01 = np.maximum(Q1, Q2)

    R00 = np.maximum(K[irange2].mean(), K[irange3].mean())
    R00 = np.maximum(R00, K[irange1].mean())

    a = K[nbins]
    K[nbins] = 0

    Qi = np.zeros(10,)
    Ri = np.zeros(10,)
    for i in range(1,11):
        irange = np.arange(nbins-i, nbins+i+1)
        Qi0 = K[irange].sum() / (2*i*tbin*len(st1)*len(st2)/T)
        Qi[i-1] = Qi0

        n = K[irange].sum()/2
        lam = R00 * i

        p = 1/2 * (1 + math.erf((n-lam)/(1e-10 + 2*lam)**.5))

        Ri[i-1] = p

    K[nbins] = a
    Q12 = np.min(Qi) / (1e-10 + np.maximum(Q00, Q01))
    R12 = np.min(Ri)

    #print('%4.2f, %4.2f, %4.2f'%(R00, Q12, R12))
    return Q12, R12, R00

def check_CCG(st1, st2=None, nbins = 500, tbin  = 1/1000):
    if st2 is None:
        st2 = st1.copy()
    K , T= compute_CCG(st1, st2, nbins = nbins, tbin = tbin)
    Q12, R12, R00 = CCG_metrics(st1, st2, K, T,  nbins = nbins, tbin = tbin)
    is_refractory    = Q12<.1  and (R12<.2  or R00<.25)
    cross_refractory = Q12<.25 and (R12<.05 or R00<.25)
    return is_refractory, cross_refractory

def refractoriness(st1, st2):
    # compute goodness of st1, st2, and both

    is_refractory = check_CCG(st1, st2)[1]
    if is_refractory:
        criterion = 1 # never split
        #print('this is refractory')
    else:
        criterion = 0
        #good_0 = check_CCG(np.hstack((st1,st2)))[0]
        #good_1 = check_CCG(st1)[0]
        #good_2 = check_CCG(st2)[0]
        #print(good_0, good_1, good_2)
        #if (good_0==1) and (good_1==0) and (good_2==0):
        #    criterion = 1 # don't split
        #    print('good cluster becomes bad')
    return criterion

def split(Xd, xtree, tstat, iclust, my_clus, verbose = True, meta = None):
    xtree = np.array(xtree)

    kk = xtree.shape[0]-1
    nc = xtree.shape[0] + 1
    valid_merge = np.ones((nc-1,), 'bool')


    for kk in range(nc-2,-1,-1):
        if not valid_merge[kk]:
            continue;

        ix1 = np.isin(iclust, my_clus[xtree[kk, 0]])
        ix2 = np.isin(iclust, my_clus[xtree[kk, 1]])

        criterion = 0
        score = np.NaN
        if criterion==0:
            # first mutation is global modularity
            if tstat[kk,0] < 0.2:
                criterion = -1


        if meta is not None and criterion==0:
            # second mutation is based on meta_data
            criterion = refractoriness(meta[ix1],meta[ix2])
            #criterion = 0
        
        if criterion==0:
            xproj, score = check_split(Xd, kk, xtree, iclust, my_clus)
            # third mutation is bimodality
            #xproj, score = check_split(Xd, kk, xtree, iclust, my_clus)
            criterion = 2 * (score <  .6) - 1

        if criterion==0:
            # fourth mutation is local modularity (not reachable)
            score = tstat[kk,-1]
            criterion = score > .15

        if verbose:
            n1,n2 = ix1.sum(), ix2.sum()
            #print('%3.0d, %6.0d, %6.0d, %6.0d, %2.2f,%4.2f, %2.2f'%(kk, n1, n2,n1+n2,
            #tstat[kk,0], tstat[kk,-1], score))

        if criterion==1:
            valid_merge[kk] = 0
            clean_tree(valid_merge, xtree, xtree[kk,0])
            clean_tree(valid_merge, xtree, xtree[kk,1])

    tstat = tstat[valid_merge]
    xtree = xtree[valid_merge]

    return xtree, tstat


def new_clusters(iclust, my_clus, xtree, tstat):

    if len(xtree)==0:
        return np.zeros_like(iclust)
         

    nc = xtree.max() + 1

    isleaf = np.zeros(2*nc-1,)
    isleaf[xtree[:,0]] = 1
    isleaf[xtree[:,1]] = 1
    isleaf[xtree[:,2]] = 0

    ind = np.nonzero(isleaf)[0]
    iclust1 = iclust.copy()
    for j in range(len(ind)):
        ix = np.isin(iclust, my_clus[ind[j]])
        iclust1[ix] = j
        xtree[xtree[:,0] == ind[j], 0] = j
        xtree[xtree[:,1] == ind[j], 1] = j


    return iclust1
