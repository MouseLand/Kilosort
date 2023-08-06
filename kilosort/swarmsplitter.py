import numpy as np
from numba import njit
import math
from kilosort.CCG import compute_CCG, CCG_metrics

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
