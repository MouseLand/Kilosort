import gc
import logging

import numpy as np
import torch
from torch import sparse_coo_tensor as coo
from scipy.sparse import csr_matrix
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.cluster.vq import kmeans
import faiss
from tqdm import tqdm 

from kilosort import hierarchical, swarmsplitter
from kilosort.utils import log_performance

logger = logging.getLogger(__name__)


def neigh_mat(Xd, nskip=10, n_neigh=30):
    # Xd is spikes by PCA features in a local neighborhood
    # finding n_neigh neighbors of each spike to a subset of every nskip spike

    # subsampling the feature matrix 
    Xsub = Xd[::nskip]

    # n_samples is the number of spikes, dim is number of features
    n_samples, dim = Xd.shape

    # n_nodes are the # subsampled spikes
    n_nodes = Xsub.shape[0]

    # search is much faster if array is contiguous
    Xd = np.ascontiguousarray(Xd)
    Xsub = np.ascontiguousarray(Xsub)

    # exact neighbor search ("brute force")
    # results is dn and kn, kn is n_samples by n_neigh, contains integer indices into Xsub
    index = faiss.IndexFlatL2(dim)   # build the index
    index.add(Xsub)    # add vectors to the index
    _, kn = index.search(Xd, n_neigh)     # actual search

    # create sparse matrix version of kn with ones where the neighbors are
    # M is n_samples by n_nodes
    dexp = np.ones(kn.shape, np.float32)    
    rows = np.tile(np.arange(n_samples)[:, np.newaxis], (1, n_neigh)).flatten()
    M   = csr_matrix((dexp.flatten(), (rows, kn.flatten())),
                   (kn.shape[0], n_nodes))

    # self connections are set to 0!
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


def assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj, device=torch.device('cuda')):
    NN = kn.shape[0]

    ij = torch.vstack((rows_neigh.flatten(), isub[kn].flatten()))
    xN = coo(ij, tones2.flatten(), (NN, nclust))
    xN = xN.to_dense()

    if lam > 0:
        tones = torch.ones(len(kj), device = device)
        tzeros = torch.zeros(len(kj), device = device)
        ij = torch.vstack((tzeros, isub))    
        kN = coo(ij, tones, (1, nclust))
    
        xN = xN - lam/m * (ki.unsqueeze(-1) * kN.to_dense()) 
    
    iclust = torch.argmax(xN, 1)

    return iclust


def assign_isub(iclust, kn, tones2, nclust, nsub, lam, m,ki,kj, device=torch.device('cuda')):
    n_neigh = kn.shape[1]
    cols = iclust.unsqueeze(-1).tile((1, n_neigh))
    iis = torch.vstack((kn.flatten(), cols.flatten()))

    xS = coo(iis, tones2.flatten(), (nsub, nclust))
    xS = xS.to_dense()

    if lam > 0:
        tones = torch.ones(len(ki), device = device)
        tzeros = torch.zeros(len(ki), device = device)
        ij = torch.vstack((tzeros, iclust))    
        kN = coo(ij, tones, (1, nclust))
        xS = xS - lam / m * (kj.unsqueeze(-1) * kN.to_dense())

    isub = torch.argmax(xS, 1)
    return isub


def Mstats(M, device=torch.device('cuda')):
    m = M.sum()
    ki = np.array(M.sum(1)).flatten()
    kj = np.array(M.sum(0)).flatten()
    ki = m * ki/ki.sum()
    kj = m * kj/kj.sum()

    ki = torch.from_numpy(ki).to(device)
    kj = torch.from_numpy(kj).to(device)
    
    return m, ki, kj


def cluster(Xd, iclust = None, kn = None, nskip = 20, n_neigh = 10, nclust = 200, 
            seed = 1, niter = 200, lam = 0, device=torch.device('cuda')):    

    if kn is None:
        kn, M = neigh_mat(Xd, nskip = nskip, n_neigh = n_neigh)

    m, ki, kj = Mstats(M, device=device)

    #Xg = torch.from_numpy(Xd).to(dev)
    Xg = Xd.to(device)
    kn = torch.from_numpy(kn).to(device)

    n_neigh = kn.shape[1]
    NN, nfeat = Xg.shape
    nsub = (NN-1)//nskip + 1

    rows_neigh = torch.arange(NN, device = device).unsqueeze(-1).tile((1,n_neigh))
    
    tones2 = torch.ones((NN, n_neigh), device = device)

    if iclust is None:
        iclust_init =  kmeans_plusplus(Xg, niter = nclust, seed = seed, device=device)
        iclust = iclust_init.clone()
    else:
        iclust_init = iclust.clone()
        
    for t in range(niter):
        # given iclust, reassign isub
        isub = assign_isub(iclust, kn, tones2, nclust , nsub, lam, m,ki,kj, device=device)

        # given mu and isub, reassign iclust
        iclust = assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj, device=device)
    
    _, iclust = torch.unique(iclust, return_inverse=True)    
    nclust = iclust.max() + 1
    isub = assign_isub(iclust, kn, tones2, nclust , nsub, lam, m,ki,kj, device=device)

    iclust = iclust.cpu().numpy()
    isub = isub.cpu().numpy()

    return iclust, isub, M, iclust_init


def kmeans_plusplus(Xg, niter = 200, seed = 1, device=torch.device('cuda')):
    #Xg = torch.from_numpy(Xd).to(dev)    
    vtot = (Xg**2).sum(1)

    n1 = vtot.shape[0]
    if n1 > 2**24:
        # Need to subsample v2, torch.multinomial doesn't allow more than 2**24
        # elements. We're just using this to sample some spikes, so it's fine to
        # not use all of them.
        n2 = n1 - 2**24   # number of spikes to remove before sampling
        remove = np.round(np.linspace(0, n1-1, n2)).astype(int)
        idx = np.ones(n1, dtype=bool)
        idx[remove] = False
        # Also need to map the indices from the subset back to indices for
        # the full tensor.
        rev_idx = idx.nonzero()[0]
        subsample = True
    else:
        subsample = False

    torch.manual_seed(seed)
    np.random.seed(seed)

    ntry = 100
    NN, nfeat = Xg.shape    
    mu = torch.zeros((niter, nfeat), device = device)
    vexp0 = torch.zeros(NN,device = device)

    iclust = torch.zeros((NN,), dtype = torch.int, device = device)
    for j in range(niter):
        v2 = torch.relu(vtot - vexp0)
        if subsample:
            isamp = rev_idx[torch.multinomial(v2[idx], ntry)]
        else:
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
    xtree, tstat = swarmsplitter.split(Xd.numpy(), xtree, tstat, iclust,
                                       my_clus, meta = st0)
    iclust1 = swarmsplitter.new_clusters(iclust, my_clus, xtree, tstat)

    return iclust1


def xy_templates(ops):
    iU = ops['iU'].cpu().numpy()
    iC = ops['iCC'][:, ops['iU']]
    #PID = st[:,5].long()
    xcup, ycup = ops['xc'][iU], ops['yc'][iU]
    xy = np.vstack((xcup, ycup))
    xy = torch.from_numpy(xy)

    iU = ops['iU'].cpu().numpy()
    iC = ops['iCC'][:, ops['iU']]    

    return xy, iC


def xy_up(ops):
    xcup, ycup = ops['xcup'], ops['ycup']
    xy = np.vstack((xcup, ycup))
    xy = torch.from_numpy(xy)
    iC = ops['iC'] 

    return xy, iC


def x_centers(ops):
    k = ops.get('x_centers', None)
    if k is not None:
        # Use this as the input for k-means, either a number of centers
        # or initial guesses.
        approx_centers = k
    else:
        # NOTE: This automated method does not work well for 2D array probes.
        #       We recommend specifying `x_centers` manually for that case.

        # Originally bin_width was set equal to `dminx`, but decided it's better
        # to not couple this behavior with that setting. A bin size of 50 microns
        # seems to work well for NP1 and 2, tetrodes, and 2D arrays. We can make
        # this a parameter later on if it becomes a problem.
        bin_width = 50
        min_x = ops['xc'].min()
        max_x = ops['xc'].max()

        # Make histogram of x-positions with bin size roughly equal to dminx,
        # with a bit of padding on either end of the probe so that peaks can be
        # detected at edges.
        num_bins = int((max_x-min_x)/(bin_width)) + 4
        bins = np.linspace(min_x - bin_width*2, max_x + bin_width*2, num_bins)
        hist, edges = np.histogram(ops['xc'], bins=bins)
        # Apply smoothing to make peak-finding simpler.
        smoothed = gaussian_filter(hist, sigma=0.5)
        peaks, _ = find_peaks(smoothed)
        # peaks are indices, translate back to position in microns
        approx_centers = [edges[p] for p in peaks]

        # Use these as initial guesses for centroids in k-means to get
        # a more accurate value for the actual centers. If there's one or none,
        # just look for one centroid.
        if len(approx_centers) <= 1: approx_centers = 1

    centers, distortion = kmeans(ops['xc'], approx_centers, seed=5330)

    # TODO: Maybe use distortion to raise warning if it seems too large?
    # "The mean (non-squared) Euclidean distance between the observations passed
    #  and the centroids generated. Note the difference to the standard definition
    #  of distortion in the context of the k-means algorithm, which is the sum of
    #  the squared distances."

    # For example, could raise a warning if this is greater than dminx*2?
    # Most probes should satisfy that criteria.

    return centers


def y_centers(ops):
    ycup = ops['ycup']
    dmin = ops['dmin']
    # TODO: May want to add the -dmin/2 in the future to center these, but
    #       this changes the results for testing so we need to wait until we can
    #       check it with simulations.
    centers = np.arange(ycup.min()+dmin-1, ycup.max()+dmin+1, 2*dmin)# - dmin/2

    return centers


def run(ops, st, tF,  mode = 'template', device=torch.device('cuda'),
        progress_bar=None, clear_cache=False):

    if mode == 'template':
        xy, iC = xy_templates(ops)
        iclust_template = st[:,1].astype('int32')
        xcup, ycup = ops['xcup'], ops['ycup']
    else:
        xy, iC = xy_up(ops)
        iclust_template = st[:,5].astype('int32')
        xcup, ycup = ops['xcup'], ops['ycup']

    dmin = ops['dmin']
    dminx = ops['dminx']
    nskip = ops['settings']['cluster_downsampling']
    ycent = y_centers(ops)
    xcent = x_centers(ops)
    nsp = st.shape[0]

    # Get positions of all grouping centers
    ycent_pos, xcent_pos = np.meshgrid(ycent, xcent)
    ycent_pos = torch.from_numpy(ycent_pos.flatten())
    xcent_pos = torch.from_numpy(xcent_pos.flatten())
    # Compute distances from templates
    center_distance = (
        (xy[0,:] - xcent_pos.unsqueeze(-1))**2
        + (xy[1,:] - ycent_pos.unsqueeze(-1))**2
        )
    # Add some randomness in case of ties
    center_distance += 1e-20*torch.rand(center_distance.shape)
    # Get flattened index of x-y center that is closest to template
    minimum_distance = torch.min(center_distance, 0).indices
    
    clu = np.zeros(nsp, 'int32')
    Wall = torch.zeros((0, ops['Nchan'], ops['settings']['n_pcs']))
    nearby_chans_empty = 0
    nmax = 0

    for kk in tqdm(np.arange(len(ycent)), miniters=20 if progress_bar else None,
                   mininterval=10 if progress_bar else None):
        for jj in np.arange(len(xcent)):
            # Get data for all templates that were closest to this x,y center.
            ii = kk + jj*ycent.size
            ix = (minimum_distance == ii)
            Xd, ch_min, ch_max, igood  = get_data_cpu(
                ops, xy, iC, iclust_template, tF, ycent[kk], xcent[jj], dmin=dmin,
                dminx=dminx, ix=ix
                )

            if ii % 10 == 0:
                log_performance(logger, header=f'Cluster center: {ii}')

            if Xd is None:
                nearby_chans_empty += 1
                continue
            elif Xd.shape[0]<1000:
                iclust = torch.zeros((Xd.shape[0],))
            else:
                if mode == 'template':
                    st0 = st[igood,0]/ops['fs']
                else:
                    st0 = None

                # find new clusters
                iclust, iclust0, M, iclust_init = cluster(Xd, nskip=nskip, lam=1,
                                                          seed=5, device=device)
                if clear_cache:
                    gc.collect()
                    torch.cuda.empty_cache()

                xtree, tstat, my_clus = hierarchical.maketree(M, iclust, iclust0)

                xtree, tstat = swarmsplitter.split(
                    Xd.numpy(), xtree, tstat,iclust, my_clus, meta=st0
                    )

                iclust = swarmsplitter.new_clusters(iclust, my_clus, xtree, tstat)

            clu[igood] = iclust + nmax
            Nfilt = int(iclust.max() + 1)
            nmax += Nfilt

            # we need the new templates here         
            W = torch.zeros((Nfilt, ops['Nchan'], ops['settings']['n_pcs']))
            for j in range(Nfilt):
                w = Xd[iclust==j].mean(0)
                W[j, ch_min:ch_max, :] = torch.reshape(w, (-1, ops['settings']['n_pcs'])).cpu()
            
            Wall = torch.cat((Wall, W), 0)

            if progress_bar is not None:
                progress_bar.emit(int((kk+1) / len(ycent) * 100))
            

    if nearby_chans_empty == len(ycent):
        raise ValueError(
            f'`get_data_cpu` never found suitable channels in `clustering_qr.run`.'
            f'\ndmin, dminx, and xcenter are: {dmin, dminx, xcup.mean()}'
        )

    if Wall.sum() == 0:
        # Wall is empty, unspecified reason
        raise ValueError(
            'Wall is empty after `clustering_qr.run`, cannot continue clustering.'
        )

    return clu, Wall


def get_data_cpu(ops, xy, iC, PID, tF, ycenter, xcenter, dmin=20, dminx=32,
                 ix=None, merge_dim=True):
    PID =  torch.from_numpy(PID).long()

    #iU = ops['iU'].cpu().numpy()
    #iC = ops['iCC'][:, ops['iU']]    
    #xcup, ycup = ops['xc'][iU], ops['yc'][iU]
    #xy = np.vstack((xcup, ycup))
    #xy = torch.from_numpy(xy)
    
    y0 = ycenter # xy[1].mean() - ycenter
    x0 = xcenter #xy[0].mean() - xcenter

    #print(dmin, dminx)
    if ix is None:
        ix = torch.logical_and(
            torch.abs(xy[1] - y0) < dmin,
            torch.abs(xy[0] - x0) < dminx
            )
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

    if merge_dim:
        Xd = torch.reshape(dd, (nspikes, -1))
    else:
        # Keep channels and features separate
        Xd = dd

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
