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


def neigh_mat(Xd, nskip=1, n_neigh=10, max_sub=25000):
    # Xd is spikes by PCA features in a local neighborhood
    # finding n_neigh neighbors of each spike to a subset of every nskip spike

    # n_samples is the number of spikes, dim is number of features
    n_samples, dim = Xd.shape

    # Downsample feature matrix by selecting every `nskip`-th spike
    Xsub = Xd[::nskip]
    n1 = Xsub.shape[0]
    # If the downsampled matrix is still larger than max_sub,
    # downsample it further by selecting `max_sub` evenly distributed spikes.
    if (max_sub is not None) and (n1 > max_sub):
        n2 = n1 - max_sub
        idx, rev_idx = subsample_idx(n1, n2)
        Xsub = Xsub[idx]
    else:
        rev_idx = None

    # n_nodes are the # subsampled spikes
    n_nodes = Xsub.shape[0]

    # search is much faster if array is contiguous
    Xd = np.ascontiguousarray(Xd)
    Xsub = np.ascontiguousarray(Xsub)

    # exact neighbor search ("brute force")
    # results is dn and kn
    # kn is n_spikes by n_neigh, contains integer indices into Xsub
    index = faiss.IndexFlatL2(dim)   # build the index
    index.add(Xsub)    # add vectors to the index
    _, kn = index.search(Xd, n_neigh)     # actual search

    # create sparse matrix version of kn with ones where the neighbors are
    # M is n_samples by n_nodes, adjacency matrix
    dexp = np.ones(kn.shape, np.float32)    
    rows = np.tile(np.arange(n_samples)[:, np.newaxis], (1, n_neigh)).flatten()
    M   = csr_matrix((dexp.flatten(), (rows, kn.flatten())),  # (data, (row,col))
                     (kn.shape[0], n_nodes))                  # (shape)

    # self connections are set to 0
    skip_idx = np.arange(0, n_samples, nskip)
    if rev_idx is not None:
        skip_idx = skip_idx[rev_idx]
    M[skip_idx, np.arange(n_nodes)] = 0

    return kn, M


def assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj, device=torch.device('cuda')):
    n_spikes = kn.shape[0]

    ij = torch.vstack((rows_neigh.flatten(), isub[kn].flatten()))
    xN = coo(ij, tones2.flatten(), (n_spikes, nclust))
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


def cluster(Xd, iclust=None, kn=None, nskip=1, n_neigh=10, max_sub=25000,
            nclust=200, seed=1, niter=200, lam=0, device=torch.device('cuda'),
            verbose=False):    

    if kn is None:
        # kn: n_spikes by n_neigh with integer indices into the spike subset
        #     used for neighbor-finding determined by nskip.
        # M:  n_spikes by nsub, adjacency matrix representation of kn.
        kn, M = neigh_mat(Xd, nskip=nskip, n_neigh=n_neigh, max_sub=max_sub)
    m, ki, kj = Mstats(M, device=device)

    if verbose:
        logger.debug(f'ki: {ki.nbytes / (2**20):.2f} MB, shape: {ki.shape}')
        logger.debug(f'kj: {kj.nbytes / (2**20):.2f} MB, shape: {kj.shape}')
        log_performance(logger, header='clustering_qr.cluster, after Mstats')

    Xg = Xd.to(device)
    kn = torch.from_numpy(kn).to(device)
    n_spikes, n_neigh = kn.shape
    nsub = M.shape[1]  # number of spikes in neighbor-finding subset

    # rows_neigh, tones2 are just used to build properly formatted indices for
    # use by assign_isub and assign_iclust
    rows_neigh = torch.arange(n_spikes, device=device).unsqueeze(-1).tile((1,n_neigh))
    tones2 = torch.ones((n_spikes, n_neigh), device=device)

    if verbose:
        logger.debug(f'Xg: {Xg.nbytes / (2**20):.2f} MB, shape: {Xg.shape}')
        logger.debug(f'kn: {kn.nbytes / (2**20):.2f} MB, shape: {kn.shape}')
        logger.debug(f'rows_neigh: {rows_neigh.nbytes / (2**20):.2f} MB')
        logger.debug(f'tones2: {tones2.nbytes / (2**20):.2f} MB')
        log_performance(logger, header='clustering_qr.cluster, after var init')

    if iclust is None:
        iclust_init =  kmeans_plusplus(Xg, niter=nclust, seed=seed, 
                                       device=device, verbose=verbose)
        iclust = iclust_init.clone()
    else:
        iclust_init = iclust.clone()
        
    for t in range(niter):
        # given iclust, reassign isub
        isub = assign_isub(iclust, kn, tones2, nclust, nsub, lam, m,
                           ki, kj,device=device)
        # given mu and isub, reassign iclust
        iclust = assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m,
                               ki, kj, device=device)
        
    if verbose:
        logger.debug(f'isub: {isub.nbytes / (2**20):.2f} MB, shape: {isub.shape}')
        log_performance(logger, header='clustering_qr.cluster, after isub loop')
    
    _, iclust = torch.unique(iclust, return_inverse=True)    
    nclust = iclust.max() + 1
    isub = assign_isub(iclust, kn, tones2, nclust , nsub, lam, m,ki,kj, device=device)

    iclust = iclust.cpu().numpy()
    isub = isub.cpu().numpy()

    return iclust, isub, M, iclust_init


def kmeans_plusplus(Xg, niter=200, seed=1, device=torch.device('cuda'), verbose=False):
    # Xg is number of spikes by number of features.
    # We are finding cluster centroids and assigning each spike to a centroid.
    vtot = torch.norm(Xg, 2, dim=1)**2

    n1 = vtot.shape[0]
    if n1 > 2**24:
        # This subsampling step is just for the candidate spikes to be considered
        # as new centroids. Sometimes need to subsample v2 since
        # torch.multinomial doesn't allow more than 2**24 elements. We're just
        # using this to sample some spikes, so it's fine to not use all of them.
        n2 = n1 - 2**24   # number of spikes to remove before sampling
        idx, rev_idx = subsample_idx(n1, n2)
        rev_idx = torch.from_numpy(rev_idx).to(device)
        subsample = True
    else:
        subsample = False

    torch.manual_seed(seed)
    np.random.seed(seed)

    ntry = 100  # number of candidate cluster centroids to test on each iteration
    n_spikes, n_features = Xg.shape
    # Need to store the spike features used for each cluster centroid (mu),
    # best variance explained so far for each spike (vexp0),
    # and the cluster assignment for each spike (iclust).
    mu = torch.zeros((niter, n_features), device = device)
    vexp0 = torch.zeros(n_spikes, device = device)
    iclust = torch.zeros((n_spikes,), dtype = torch.int, device = device)

    if verbose:
        log_performance(logger, header='clustering_qr.kpp, after var init')

    # On every iteration we choose one new centroid to keep.
    # We track how well n centroids so far explain each spike.
    # We ask, if we were to add another centroid, which spikes would that
    # increase the explained variance for and by how much?
    # We use ntry candidates on each iteration.
    for j in range(niter):
        # v2 is the un-explained variance so far for each spike
        v2 = torch.relu(vtot - vexp0)

        # We sample ntry new candidate centroids based on how much un-explained variance they have
        # more unexplained variance makes it more likely to be selected
        # Only one of these candidates will be added this iteration. 
        if subsample:
            isamp = rev_idx[torch.multinomial(v2[idx], ntry)]
        else:
            isamp = torch.multinomial(v2, ntry)

        try:
            # The new centroids to be tested, sampled from the spikes in Xg.
            Xc = Xg[isamp]
            # Variance explained for each spike for the new centroids.
            vexp = 2 * Xg @ Xc.T - (Xc**2).sum(1)
            # Difference between variance explained for new centroids
            # and best explained variance so far across all iterations.
            # This gets relu-ed, since only the positive increases will actually
            # re-assign a spike to this new cluster
            dexp = torch.relu(vexp - vexp0.unsqueeze(1))
            # Sum all positive increases to determine additional explained variance
            # for each candidate centroid.
            vsum = dexp.sum(0)
            # Pick the candidate which increases explained variance the most 
            imax = torch.argmax(vsum)

            # For that centroid (Xc[imax]), determine which spikes actually get
            # more variance from it
            ix = dexp[:, imax] > 0

            iclust[ix] = j    # assign new cluster identity
            mu[j] = Xc[imax]  # spike features used as centroid for cluster j
            # Update variance explained for the spikes assigned to cluster j
            vexp0[ix] = vexp[ix, imax]

            # Delete large variables between iterations
            # to prevent excessive memory reservation.
            del(vexp)
            del(dexp)

        except torch.cuda.OutOfMemoryError:
            logger.debug(f"OOM in kmeans_plus_plus iter {j}, nsp: {Xg.shape[0]}, "
                         f"Xg size: {Xg.nbytes / (2**20):.2f} MB.")
            raise

    if verbose:
        log_performance(logger, header='clustering_qr.kpp, after loop')

    # NOTE: For very large datasets, we may end up needing to subsample Xg.
    # If the clustering above is done on a subset of Xg,
    # then we need to assign all Xgs here to get an iclust 
    # for ii in range((len(Xg)-1)//nblock +1):
    #     vexp = 2 * Xg[ii*nblock:(ii+1)*nblock] @ mu.T - (mu**2).sum(1)
    #     iclust[ii*nblock:(ii+1)*nblock] = torch.argmax(vexp, dim=-1)

    return iclust


def subsample_idx(n1, n2):
    """Get boolean mask and reverse mapping for evenly distributed subsample.
    
    Parameters
    ----------
    n1 : int
        Size of index. Index is assumed to be sequential and not contain any
        missing values (i.e. 0, 1, 2, ... n1-1).
    n2 : int
        Number of indices to remove to create a subsample. Removed indices are
        evenly spaced across 
    
    Returns
    -------
    idx : np.ndarray
        Boolean mask, True for indices to be included in the subset.
    rev_idx : np.ndarray
        Map between subset indices and their position in the original index.

    Examples
    --------
    >>> subsample_idx(6, 3)
    array([False,  True, False,  True,  True, False], dtype=bool),
    array([1, 3, 4], dtype=int64)

    """

    remove = np.round(np.linspace(0, n1-1, n2)).astype(int)
    idx = np.ones(n1, dtype=bool)
    idx[remove] = False
    # Also need to map the indices from the subset back to indices for
    # the full tensor.
    rev_idx = idx.nonzero()[0]

    return idx, rev_idx


def xy_templates(ops):
    iU = ops['iU'].cpu().numpy()
    iC = ops['iCC'][:, ops['iU']]
    #PID = st[:,5].long()
    xcup, ycup = ops['xc'][iU], ops['yc'][iU]
    xy = np.vstack((xcup, ycup))
    xy = torch.from_numpy(xy)

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


def get_nearest_centers(xy, xcent, ycent):
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

    return minimum_distance, xcent_pos, ycent_pos


def run(ops, st, tF, mode='template', device=torch.device('cuda'),
        progress_bar=None, clear_cache=False, verbose=False):

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
    n_neigh = ops['settings']['cluster_neighbors']
    max_sub = ops['settings']['max_cluster_subset']
    seed = ops['settings']['cluster_init_seed']
    ycent = y_centers(ops)
    xcent = x_centers(ops)
    nsp = st.shape[0]
    nearest_center, _, _ = get_nearest_centers(xy, xcent, ycent)
    total_centers = np.unique(nearest_center).size
    
    clu = np.zeros(nsp, 'int32')
    Wall = torch.zeros((0, ops['Nchan'], ops['settings']['n_pcs']))
    Nfilt = None
    nearby_chans_empty = 0
    nmax = 0
    prog = tqdm(np.arange(len(xcent)), miniters=20 if progress_bar else None,
                mininterval=10 if progress_bar else None)
    t = 0
    v = False
    
    try:
        for jj in prog:
            for kk in np.arange(len(ycent)):
                # Get data for all templates that were closest to this x,y center.
                ii = kk + jj*ycent.size
                if ii not in nearest_center:
                    # No templates are nearest to this center, skip it.
                    continue
                else:
                    t += 1
                ix = (nearest_center == ii)
                ntemp = ix.sum()

                v = False
                if t % 10 == 0:
                    log_performance(
                        logger,
                        header=f'Cluster center: {ii} ({t}/{total_centers})'
                        )
                    if verbose:
                        v = True

                Xd, igood, ichan = get_data_cpu(
                    ops, xy, iC, iclust_template, tF, ycent[kk], xcent[jj],
                    dmin=dmin, dminx=dminx, ix=ix,
                    )
                if Xd is None:
                    nearby_chans_empty += 1
                    continue

                logger.debug(f'Center {ii} | Xd shape: {Xd.shape} | ntemp: {ntemp}')
                if verbose and Xd.nelement() > 10**8:
                    logger.info(f'Resetting cuda memory stats for Center {ii}')
                    if device == torch.device('cuda'):
                        torch.cuda.reset_peak_memory_stats(device)
                    v = True
                if Xd.shape[0] < 1000:
                    iclust = torch.zeros((Xd.shape[0],))
                else:
                    if mode == 'template':
                        st0 = st[igood,0]/ops['fs']
                    else:
                        st0 = None

                    # find new clusters
                    iclust, iclust0, M, _ = cluster(
                        Xd, nskip=nskip, n_neigh=n_neigh, max_sub=max_sub,
                        lam=1, seed=seed, device=device, verbose=v
                        )

                    if clear_cache:
                        if v:
                            log_performance(logger, header='clustering_qr before gc')
                        gc.collect()
                        torch.cuda.empty_cache()
                        if v:
                            log_performance(logger, header='clustering_qr after gc')

                    xtree, tstat, my_clus = hierarchical.maketree(M, iclust, iclust0)

                    xtree, tstat = swarmsplitter.split(
                        Xd.numpy(), xtree, tstat,iclust, my_clus, meta=st0
                        )

                    iclust = swarmsplitter.new_clusters(iclust, my_clus, xtree, tstat)

                if v:
                    log_performance(logger, header='clustering_qr.run, after iclust')

                clu[igood] = iclust + nmax
                Nfilt = int(iclust.max() + 1)
                nmax += Nfilt

                # we need the new templates here         
                W = torch.zeros((Nfilt, ops['Nchan'], ops['settings']['n_pcs']))
                for j in range(Nfilt):
                    w = Xd[iclust==j].mean(0)
                    W[j, ichan, :] = torch.reshape(w, (-1, ops['settings']['n_pcs'])).cpu()
                
                Wall = torch.cat((Wall, W), 0)

                if progress_bar is not None:
                    progress_bar.emit(int((kk+1) / len(ycent) * 100))
    except:
        logger.exception(f'Error in clustering_qr.run on center {ii}')
        logger.debug(f'Xd shape: {Xd.shape}')
        logger.debug(f'Nfilt: {Nfilt}')
        logger.debug(f'num spikes: {nsp}')
        try:
            logger.debug(f'iclust shape: {iclust.shape}')
        except UnboundLocalError:
            logger.debug('iclust not yet assigned')
            pass
        raise

    if nearby_chans_empty == total_centers:
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

    if ix is None:
        ix = torch.logical_and(
            torch.abs(xy[1] - y0) < dmin,
            torch.abs(xy[0] - x0) < dminx
            )
    igood = ix[PID].nonzero()[:,0]

    if len(igood) == 0:
        return None, None, None

    pid = PID[igood]
    data = tF[igood]
    nspikes, nchanraw, nfeatures = data.shape
    ichan, imap = torch.unique(iC[:, ix], return_inverse=True)
    nchan = ichan.nelement()

    dd = torch.zeros((nspikes, nchan, nfeatures))
    for k,j in enumerate(ix.nonzero()[:,0]):
        ij = torch.nonzero(pid==j)[:, 0]
        dd[ij.unsqueeze(-1), imap[:,k]] = data[ij]

    if merge_dim:
        Xd = torch.reshape(dd, (nspikes, -1))
    else:
        # Keep channels and features separate
        Xd = dd

    return Xd, igood, ichan


def assign_clust(rows_neigh, iclust, kn, tones2, nclust):    
    n_spikes = len(iclust)

    ij = torch.vstack((rows_neigh.flatten(), iclust[kn].flatten()))
    xN = coo(ij, tones2.flatten(), (n_spikes, nclust))
    
    xN = xN.to_dense() 
    iclust = torch.argmax(xN, 1)

    return iclust

def assign_iclust0(Xg, mu):
    vv = Xg @ mu.T
    nm = (mu**2).sum(1)
    iclust = torch.argmax(2*vv-nm, 1)
    return iclust
