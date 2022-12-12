import torch, os
from kilosort import preprocessing
dev = torch.device('cuda:0')
import numpy as np

def avg_wav(Wsub, nn, ops, ibatch, st_i, clu, Nfilt):
    nt = ops['nt']
    NT = ops['NT']

    X = preprocessing.load_transform(ops['filename'], ibatch, ops, fwav = ops['fwav'], 
                                Wrot = ops['Wrot'], dshift = ops['dshift'])


    ix = (st_i - NT * ibatch) * (st_i - NT * (1+ibatch)) < 0
    st_sub = st_i[ix] + nt - NT * ibatch
    st_sub = torch.from_numpy(st_sub).to(dev)


    tiwave = torch.arange(nt, device=dev)
    xsub = X[:, st_sub.unsqueeze(-1) + tiwave] @ ops['wPCA'].T
    nsp = xsub.shape[1]
    M = torch.zeros((Nfilt, nsp), dtype=torch.float, device = dev)
    M[clu[ix], torch.arange(nsp)] = 1
    
    Wsub += torch.einsum('ijk, lj -> lki', xsub, M)
    nn += M.sum(1)

    return Wsub, nn

def clu_ypos(ops, st_i, clu):
    Nfilt = clu.max()+1
    Wsub = torch.zeros((Nfilt, ops['nwaves'], ops['Nchan']), device = dev)
    nn   = torch.zeros((Nfilt, ), device = dev)

    for ibatch in range(0, ops['Nbatches'], 10):
        Wsub, nn = avg_wav(Wsub, nn, ops, ibatch, st_i, clu, Nfilt)

    Wsub = Wsub / nn.unsqueeze(-1).unsqueeze(-1)
    #Wsub = Wsub / nn.unsqueeze(-1).unsqueeze(-1)
    Wsub = Wsub.cpu().numpy()

    ichan = np.argmax((Wsub**2).sum(1), -1)
    yclu = ops['yc'][ichan]

    return yclu, Wsub

def nmatch(ss0, ss):
    i = 0
    j = 0
    dt = 3
    n0 = 0

    ntmax = len(ss0)
    is_matched  = np.zeros(len(ss), 'bool')
    is_matched0 = np.zeros(len(ss0), 'bool')

    while i<len(ss):
        while j+1<ntmax and ss0[j+1] < ss[i]+dt:
            j += 1

        if np.abs(ss0[j] - ss[i]) <=dt:
            n0 += 1
            is_matched[i] = 1
            is_matched0[j] = 1

        i+= 1
    return n0, is_matched, is_matched0

def match_neuron(kk, clu, yclu, st_i, clu0, yclu0, st0_i):
    ss = st_i[clu==kk]
    isort = np.argsort(np.abs(yclu[kk] - yclu0))

    fmax = 0
    miss = 0
    fpos = 0
    for j in range(20):
        ss0 = st0_i[clu0==isort[j]]

        if len(ss0) ==0:
            continue
        
        n0, is_matched, is_matched0 = nmatch(ss0, ss)

        
        #fmax_new = n0 / (len(ss) + len(ss0) - n0)
        fmax_new = 1 - np.maximum(0, 1 - n0/len(ss)) - np.maximum(0, 1 - n0/len(ss0))

        if fmax_new > fmax:
            miss = np.maximum(0, 1 - n0/len(ss))
            fpos = np.maximum(0, 1 - n0/len(ss0))

            fmax = fmax_new

    return fmax, miss, fpos


def compare_recordings(st_gt, clu_gt, yclu_gt, st_new, clu_new, yclu_new):
    NN = len(yclu_gt)

    fmax = np.zeros(NN,)
    fmiss = np.zeros(NN,)
    fpos = np.zeros(NN,)

    for kk in range(NN):
        fmax[kk], fmiss[kk], fpos[kk] = match_neuron(kk, clu_gt, yclu_gt, st_gt, clu_new, yclu_new, st_new)

    return fmax, fmiss, fpos

def load_GT(ops, gt_path, toff = 20, nmax = 600):
    #gt_path = os.path.join(ops['data_folder'] , "sim.imec0.ap_params.npz")
    dd = np.load(gt_path, allow_pickle = True)

    st_gt = dd['st'].astype('int64')
    clu_gt = dd['cl'].astype('int64')

    ix = clu_gt<nmax
    st_gt  = st_gt[ix]
    clu_gt = clu_gt[ix]

    yclu_gt, Wsub = clu_ypos(ops, st_gt - toff, clu_gt.astype('int64'))
    mu_gt = (Wsub**2).sum((1,2))**.5

    unq_clu, nsp = np.unique(clu_gt, return_counts = True)
    if np.abs(np.diff(unq_clu) - 1).sum()>1e-5:
        print('error, some ground truth units are missing')

    return st_gt, clu_gt, yclu_gt, mu_gt, Wsub, nsp


def convert_ks_output(ops, st, clu, toff = 20):
    st = st[:,0].astype('int64')        
    yclu, Wsub    = clu_ypos(ops, st-toff, clu)

    return st, clu, yclu, Wsub


def load_phy(fpath, ops):
    st_new  = np.load(os.path.join(fpath,  "spike_times.npy")).astype('int64')
    clu_new = np.load(os.path.join(fpath ,"spike_clusters.npy")).astype('int64')
    
    yclu_new, Wsub = clu_ypos(ops, st_new - 20, clu_new)

    return st_new, clu_new, yclu_new, Wsub