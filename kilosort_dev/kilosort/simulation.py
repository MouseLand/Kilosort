import numpy as np 
import os, time 
import matplotlib.pyplot as plt 
import torch 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.sparse import csr_matrix 
from scipy.ndimage import gaussian_filter1d
from tqdm import trange
import pandas as pd

from kilosort.datashift import kernel2D
from kilosort import io

def preprocess_wfs_sts(filename, y_chan, x_chan, y_chan_up=None, x_chan_up=None,
                       ups=10, device=torch.device('cuda')):
    if y_chan_up is None:
        y_chan_up = y_chan
        x_chan_up = x_chan

    ### waveforms and spike trains from data
    f = np.load(filename, allow_pickle=True)
    sts_all = f['sts_all']
    cls_all = f['cls_all']
    wfs_all = f['wfs_all']
    contaminations_all = f['contaminations_all']
    wfs_x_all = f['wfs_x_all']
    nst_all = f['nst_all']

    ### take all waveforms with at least 50 spikes per drift bin
    contaminations = np.concatenate(contaminations_all, axis=0)
    wfs = np.concatenate(wfs_all, axis=0)
    wfs_x = np.concatenate(wfs_x_all, axis=0)
    nsts = np.concatenate(nst_all, axis=0)
    idr = np.nonzero((nsts>50).sum(axis=1)==nsts.shape[1])[0]
    contaminations = contaminations[idr]
    wfs = wfs[idr]
    wfs_x = wfs_x[idr]
    wfs /= ((wfs**2).sum(axis=(2,3), keepdims=True)**0.5).mean(axis=1, keepdims=True)

    ### remove low firing spike trains
    sts = np.zeros(0, 'uint64')
    cls = np.zeros(0, 'uint32')
    ntm = np.array([sts_all[i].max() for i in range(len(sts_all))]).min()
    i0 = 0
    min_fr = 0.5 # min firing rate of 0.5 Hz
    for i in range(len(sts_all)):
        n_max = np.nonzero(sts_all[i] > ntm-1)[0][0]
        st0 = sts_all[i][:n_max]
        cl0 = cls_all[i][:n_max]
        cu, cc = np.unique(cl0, return_counts=True)
        cc = cc.astype('float32')
        cc /= st0[-1]
        cc *= 30000
        ids = cu[np.nonzero(cc>=min_fr)[0]]
        ix = np.isin(cl0, ids)
        clsx = np.unique(cl0[ix], return_inverse=True)[1]
        stsx = st0[ix]
        sts = np.append(sts, stsx)
        cls = np.append(cls, i0 + clsx)
        i0 += clsx.max() + 1
    isort = sts.argsort()
    cls = cls[isort]
    sts = sts[isort]
    cc = np.unique(cls, return_counts=True)[1].astype('float32')
    cc /= sts[-1]
    cc *= 30000
    print('# of spike trains: ', len(cc))
    plt.hist(cc, 100);
    plt.show()

    ### denoise waveforms
    # use 3-component PC reconstruction of waveforms
    # keep waveforms that don't vary too much across drift
    npc = 3
    wt = torch.from_numpy(wfs).to(device)
    wt = wt.reshape(wt.shape[0]*wt.shape[1], wt.shape[2], wt.shape[3])
    u,s,v = torch.svd(wt)
    wtt = torch.einsum("nij,nkj->nik", u[:,:,:npc] * s[:,:npc].unsqueeze(1), v[:,:,:npc])
    wfs = wtt.cpu().numpy()
    wfs = wfs.reshape(-1, 20, wfs.shape[-2], wfs.shape[-1])
    wfs /= ((wfs**2).sum(axis=(2,3), keepdims=True)**0.5).mean(axis=1, keepdims=True)

    nd = ((wfs[:, 0, :, 4:] - wfs[:, -1, :, :-4])**2).sum(axis=(-2,-1))
    na = (wfs[:, 0, :, 4:]**2).sum(axis=(-2,-1))
    igood = np.nonzero(nd / na < 0.5)[0]
    print('# of good waveforms: ', len(igood))
    wfs = wfs[igood]
    wfs_x = wfs_x[igood]
    contaminations = contaminations[igood]

    ### smooth and upsample waveforms
    # smooth across drift positions to reduce variability in waveform shape across drift
    # upsample to allow 10x more drift positions
    sig_interp = 20
    lam = 1e-2
    wfs_smoothed = np.zeros((wfs.shape[0], wfs.shape[1]*ups, *wfs.shape[2:]),'float32')
    for ic in range(4):
        iwc = wfs_x == ic
        nd,nt,nc = wfs.shape[-3:]
        
        xpu = np.vstack((y_chan_up[ic : ic+nc], x_chan_up[ic : ic+nc])).T
        zup = np.arange(0,nd*2,2./ups)[:,np.newaxis,np.newaxis]
        zup = np.concatenate((zup, np.zeros((nd*ups,1,1))), axis=-1)
        xup = (np.tile(xpu[np.newaxis,:], (nd*ups,1,1)) + zup).reshape(-1, 2)

        xp = np.vstack((y_chan[ic : ic+nc], x_chan[ic : ic+nc])).T
        zp = np.arange(0,nd*2,2)[:,np.newaxis,np.newaxis]
        zp = np.concatenate((zp, np.zeros((nd,1,1))), axis=-1)
        xp = (np.tile(xp[np.newaxis,:], (nd,1,1)) + zp).reshape(-1, 2)    
        
        Kyx = kernel2D(xup, xp, sig_interp)
        Kxx = kernel2D(xp, xp, sig_interp)
        Kyx = torch.from_numpy(Kyx).float().to(device)
        Kxx = torch.from_numpy(Kxx).float().to(device)
        iKxx = torch.linalg.inv(Kxx + lam * torch.eye(Kxx.shape[0], device=device))
        M = Kyx @ iKxx

        wf0 = torch.from_numpy(wfs[iwc]).float().to(device)
        wf0 = wf0.transpose(3,2).reshape(-1,nd*nc, nt)
        wf0s = torch.einsum('nj,ijk->ink', M, wf0) 
        wf0s = wf0s.reshape(-1, nd*ups, nc, nt).transpose(3,2)
        wfs_smoothed[iwc] = wf0s.cpu().numpy()
    wfs_smoothed /= ((wfs_smoothed**2).sum(axis=(2,3), keepdims=True)**0.5).mean(axis=1, keepdims=True)
    wfs = wfs_smoothed
    
    return wfs, wfs_x, contaminations, sts, cls


def generate_background(NT, fs=30000, device=torch.device('cuda')):
    #NT = 3000000
    nf = np.arange(NT, dtype = 'float32')
    nf = np.minimum(nf, NT - nf)
    nf = torch.from_numpy(nf).to(device)

    ff = nf
    ff[ff < NT*300/fs] = 10000000
    ff = (ff.mean() / (1+ff.unsqueeze(-1).unsqueeze(-1)))**.5

    X = torch.zeros((NT, 96, 4))

    for j in range(12):
        Xn = torch.randn(NT, 8, 4, device = device)
        Xfft = torch.fft.fft(Xn, dim = 0)
        Xfft = 100 * Xfft / torch.abs(Xfft) * ff
        Xn = torch.fft.ifft(Xfft, dim = 0)
        Xn = torch.real(Xn)
        X[:, 8*j:8*(j+1), :] = Xn.cpu()

    X /= X.std()
    X *= 0.95 * 0.8
    X = X.reshape(X.shape[0], -1)
    #X = torch.linalg.solve(whiten_mat.cpu(), X.T).T
    return X

def generate_spikes(st, cl, wfs, wfs_x, contaminations,
                    n_sim=200, n_noise=1000, n_batches=500, tsig=50, 
                    batch_size=60000, twav_min=50, drift=True,
                    drift_range=5, drift_seed=0, fast=False, step=False,
                    n_batches_sim=None, ups=10):
    """ simulate spikes using 
        - random waveforms from wfs with x-pos wfs_x 
        - spike train stats from (st, cl) 
        - drift with smoothing constant of tsig across batches 
        - first n_sim neurons are good neurons, next n_noise neurons are MUA / noise
    WARNING: requires RAM for full simulation as written

    """
    
    n_bins, n_twav, nc = wfs.shape[1:] 
    st_sim = np.zeros(0, 'uint64')
    cl_sim = np.zeros(0, 'uint32')
    wf_sim = np.zeros((n_sim + n_noise, n_bins, n_twav, nc), 'float32')
    wfid_sim = np.zeros(n_sim + n_noise, 'uint32')
    cb_sim = np.zeros(n_sim + n_noise, 'int')

    # random amplitudes
    # for good neurons
    amp_sim = np.random.exponential(scale=1., size=(n_sim,)) * 7. + 10.
    # for MUA / noise
    if n_noise > 0:
        amp_sim = np.append(amp_sim, 
                            np.random.rand(n_noise) * 6. + 4.,
                            axis=0)

    # create good neurons
    n_time = st.max() / 30000.

    n_time_sim = n_batches * batch_size
    n_max = np.nonzero(st > n_time_sim-1)[0][0]
    good_neurons = np.nonzero(contaminations < 0.1)[0]
    print('creating good neurons')
    for i in trange(n_sim):
        # random spike train, waveform and channel per neuron
        sp_rand = np.random.randint(cl.max()+1)
        wf_rand = np.random.randint(len(good_neurons))
        # pad to avoid best channel on first 4 or last 4 channels
        cb_rand = np.random.randint(96-2) + 1
        frac_rand = np.random.beta(1, 3) * 0.8 + 0.2

        # spike train
        spi = st[cl == sp_rand].copy()
        isi = np.diff(spi)
        isi = isi[np.random.permutation(len(isi))]
        spi = np.cumsum(isi)
        if len(spi) > 0:
            n_max = np.nonzero(spi > n_time_sim-1)[0][0] if spi[-1] > n_time_sim-1 else len(spi)
            spi = spi[:n_max]
            fr = (len(spi) / n_time_sim) * 30000
            if fr > 5:
                spi = spi[np.random.rand(len(spi)) < frac_rand]

            # waveform
            wfi = wfs[good_neurons[wf_rand]]
            cbi = cb_rand * 4 + wfs_x[good_neurons[wf_rand]]

            st_sim = np.append(st_sim, spi, axis=0)
            cl_sim = np.append(cl_sim, i * np.ones(len(spi), 'uint32'), axis=0)
            wf_sim[i] = wfi
            cb_sim[i] = cbi
            wfid_sim[i] = good_neurons[wf_rand]
        else:
            i -= 1

    # create MUA / noise
    if n_noise > 0:
        bad_neurons = np.nonzero(contaminations >= 0.1)[0]
        _, frs = np.unique(cl, return_counts=True)
        frs = frs.astype('float32')
        frs /= n_time
        print('creating MUA / noise')
        for i in trange(n_noise):
            # random firing rate, waveform and channel per neuron
            fr_rand = np.random.randint(len(frs))
            wf_rand = np.random.randint(len(bad_neurons))
            cb_rand = np.random.randint(96)
            frac_rand = np.random.beta(1, 3) * 0.8 + 0.2

            # spike train
            fr = frs[fr_rand]
            isi = (30000 * np.random.exponential(scale=1./fr, size=int(np.ceil((n_time_sim * fr / 30000))))).astype('uint64')
            spi = np.cumsum(isi)
            if fr > 5:
                spi = spi[np.random.rand(len(spi)) < frac_rand]

            # waveform
            wfi = wfs[bad_neurons[wf_rand]]
            cbi = cb_rand * 4 + wfs_x[bad_neurons[wf_rand]]

            st_sim = np.append(st_sim, spi, axis=0)
            cl_sim = np.append(cl_sim, (n_sim + i) * np.ones(len(spi), 'uint32'), axis=0)
            wf_sim[n_sim + i] = wfi
            cb_sim[n_sim + i] = cbi
            wfid_sim[n_sim + i] = bad_neurons[wf_rand]
        
    # re-sort by time
    ii = st_sim.argsort()
    cl_sim = cl_sim[ii]
    st_sim = st_sim[ii]

    # remove spikes after max recording time
    n_max = np.nonzero(st_sim > n_time_sim-1)[0][0] if st_sim[-1] > n_time_sim-1 else len(st_sim)
    n_min = np.nonzero(st_sim < n_twav)[0][-1] + 1
    st_sim = st_sim[n_min : n_max]
    cl_sim = cl_sim[n_min : n_max]
    print(len(st_sim))
    
    # amplitude multiply waveforms
    wf_sim *= amp_sim[:,np.newaxis,np.newaxis,np.newaxis]
    
    np.random.seed(drift_seed)
    # overall drift
    if drift:
        slow_n = n_batches // 10 if fast else n_batches

        drift_sim = np.random.randn(slow_n)
        drift_sim = gaussian_filter1d(drift_sim, tsig)
        drift_sim -= drift_sim.min()
        drift_sim /= drift_sim.max()
        drift_sim *= drift_range

        # drift across probe (9 positions)
        drift_chan = np.random.randn(slow_n, 9)
        drift_chan = gaussian_filter1d(drift_chan, tsig, axis=0)
        drift_chan -= drift_chan.min(axis=0) 
        drift_chan /= drift_chan.max(axis=0)
        drift_chan *= 5 - (20-drift_range)/6
        drift_chan = gaussian_filter1d(drift_chan*4, 2, axis=-1)
        drift_chan += drift_sim[:,np.newaxis]
        drift_chan -= drift_chan.min() 
        drift_chan /= drift_chan.max()
        drift_chan *= drift_range
        drift_chan += 10 - drift_range / 2

        if fast: 
            fast_drift = np.zeros(n_batches)
            fast_drift[np.random.permutation(n_batches)[:300]] = 1 
            t = np.arange(20)
            kernel = np.exp(-t/2.5) - np.exp(-t)
            kernel /= kernel.max()
            fast_drift = convolve1d(fast_drift, kernel)
            fast_drift *= 5
            drift_chan = np.tile(drift_chan[:,np.newaxis], (1,10,1))
            drift_chan = drift_chan.reshape(n_batches, -1)
            drift_chan -= 5
            drift_chan += fast_drift[:,np.newaxis]
        elif step:
            drift_chan -= 7.5
            drift_chan[n_batches//2:] += 15

        plt.plot(drift_chan);
        plt.show()
    else:
        drift_chan = 10 * np.ones((n_batches, 9), 'float32')

    # upsample to get drift for all channels
    f = interp1d(np.linspace(0, 383, drift_chan.shape[1]+2)[1:-1], drift_chan, axis=-1, fill_value='extrapolate')
    drift_sim_ups = f(np.arange(0,384))
    drift_sim_ups = np.floor(drift_sim_ups * ups).astype(int)
    print(drift_sim_ups.min(), drift_sim_ups.max())
    print('created drift')

    # create spikes
    n_batches = int(np.ceil((st_sim.max() + n_twav) / batch_size)) if n_batches_sim is None else n_batches_sim
    data = np.zeros((batch_size * n_batches, 385), 'float32')
    data_mua = None if n_batches_sim is None else np.zeros((batch_size * n_batches, 385), 'float32')
    st_max = batch_size * n_batches 
    cl_sim = cl_sim[st_sim < st_max - n_twav]
    st_sim = st_sim[st_sim < st_max - n_twav]
    st_batch = np.floor(st_sim / batch_size).astype('int') # batch of each spike
    n_splits = 3 if drift_range>0 else 10
    tic=time.time()
    # loop over neurons
    print('creating data')
    for i in trange(n_sim + n_noise):
        # waveform and channel for waveform
        wfi = wf_sim[i]
        ic0 = cb_sim[i] - nc//2
        iw0 = max(0, -ic0)
        ic1 = min(384, ic0+nc)
        ic0 = max(0, ic0)
        iw1 = ic1 - ic0 + iw0
        # spikes from neuron            
        is0 = cl_sim == i
        sti = st_sim[is0]
        drifti = drift_sim_ups[st_batch[is0], cb_sim[i]]
        # loop over z-positions
        for d in range(n_bins):
            ist = drifti == d
            if ist.sum() > 0:
                inds = np.nonzero(ist)[0]
                #print(len(inds))
                for k in range(n_splits):
                    ki = np.arange(k, len(inds), n_splits)
                    # indices for spike waveform
                    stb = sti[inds[ki]] - int(twav_min)
                    stb = stb[:,np.newaxis].astype(int) + np.arange(0, n_twav, 1, int)
                    if i < n_sim or data_mua is None:
                        data[stb, ic0:ic1] += wfi[d][:, iw0:iw1]
                    else:
                        data_mua[stb, ic0:ic1] += wfi[d][:, iw0:iw1]

    return data, data_mua, st_sim, cl_sim, amp_sim, wf_sim, cb_sim, wfid_sim, drift_sim_ups

def create_simulation(filename, st, cl, wfs, wfs_x, contaminations,
                      n_sim=100, n_noise=1000, n_batches=500,
                      batch_size=60000, tsig=50, tpad=100, n_chan_bin=385, drift=True,
                      drift_range=5, drift_seed=0, fast=False, step=False,
                      ups=10, whiten_mat = None,
                             ):
    """ simulate neuropixels 3B probe recording """

    # simulate spikes
    out = generate_spikes(st, cl, wfs, wfs_x, 
                        contaminations, n_sim=n_sim, n_noise=n_noise, 
                        n_batches=n_batches, tsig=tsig, 
                        batch_size=batch_size, drift=drift, 
                        fast=fast, step=step,
                        drift_range=drift_range, drift_seed=drift_seed,
                        ups=ups)
    data, data_mua, st_sim, cl_sim, amp_sim, wf_sim, cb_sim, wfid_sim, drift_sim_ups = out

    print(f'created dataset of size {data.shape}')
    
    # simulating background
    tweight = torch.arange(0, tpad, dtype=torch.float32)
    tweight_flip = tweight.flip(dims=(0,))
    tweight /= tweight + tweight_flip
    tweight = tweight.unsqueeze(1)
    tweight_flip = tweight.flip(dims=(0,))
    noise_init = generate_background(batch_size + tpad)
    noise_b = torch.zeros((batch_size, n_chan_bin))
    whiten_mat = torch.from_numpy(whiten_mat)

    # writing binary
    with io.BinaryRWFile(filename, n_chan_bin, NT=batch_size, 
                         write=True, n_samples=data.shape[0]) as bfile:
        for ibatch in np.arange(0, n_batches):
            noise_next = generate_background(batch_size + tpad)
            noise_pad = tweight_flip * noise_init[-tpad:] +  tweight * noise_next[:tpad]
            noise = torch.cat((noise_pad, noise_next[tpad : -tpad]), axis=0)
            noise_b[:, :384] = noise
            X = torch.from_numpy(data[ibatch * batch_size : min(n_batches * batch_size, (ibatch+1) * batch_size)]).float()
            
            X = (noise_b + X)

            if whiten_mat is not None:
                X[:,:384] = torch.linalg.solve(whiten_mat, X[:,:384].T).T

            X = X.numpy()
            to_write = np.clip(20 * X, -2**15 + 1, 2**15 - 1).astype('int16')
            #to_write = np.clip(X, -2**15 + 1, 2**15 - 1).astype('int16')

            if ibatch%100==0:
                print(f'writing batch {ibatch} out of {n_batches}')
            bfile[ibatch * batch_size : min(n_batches * batch_size, (ibatch+1) * batch_size)] = X
            noise_next = noise_init

    # create meta file for probe
    metaname = os.path.splitext(filename)[0] + '.meta'
    meta_dict = {'fileSizeBytes': os.path.getsize(filename),
                'typeThis': 'imec',
                'imSampRate': 30000,
                'imAiRangeMax': 0.6,
                'imAiRangeMin': -0.6,
                'nSavedChans': 385,
                'snsApLfSy': '384,0,1',
                'snsSaveChanSubset': '0:383,768',
                '~imroTbl': '(0,384)(0 0 0 500 250 1)(1 0 0 500 250 1)(2 0 0 500 250 1)(3 0 0 500 250 1)(4 0 0 500 250 1)(5 0 0 500 250 1)(6 0 0 500 250 1)(7 0 0 500 250 1)(8 0 0 500 250 1)(9 0 0 500 250 1)(10 0 0 500 250 1)(11 0 0 500 250 1)(12 0 0 500 250 1)(13 0 0 500 250 1)(14 0 0 500 250 1)(15 0 0 500 250 1)(16 0 0 500 250 1)(17 0 0 500 250 1)(18 0 0 500 250 1)(19 0 0 500 250 1)(20 0 0 500 250 1)(21 0 0 500 250 1)(22 0 0 500 250 1)(23 0 0 500 250 1)(24 0 0 500 250 1)(25 0 0 500 250 1)(26 0 0 500 250 1)(27 0 0 500 250 1)(28 0 0 500 250 1)(29 0 0 500 250 1)(30 0 0 500 250 1)(31 0 0 500 250 1)(32 0 0 500 250 1)(33 0 0 500 250 1)(34 0 0 500 250 1)(35 0 0 500 250 1)(36 0 0 500 250 1)(37 0 0 500 250 1)(38 0 0 500 250 1)(39 0 0 500 250 1)(40 0 0 500 250 1)(41 0 0 500 250 1)(42 0 0 500 250 1)(43 0 0 500 250 1)(44 0 0 500 250 1)(45 0 0 500 250 1)(46 0 0 500 250 1)(47 0 0 500 250 1)(48 0 0 500 250 1)(49 0 0 500 250 1)(50 0 0 500 250 1)(51 0 0 500 250 1)(52 0 0 500 250 1)(53 0 0 500 250 1)(54 0 0 500 250 1)(55 0 0 500 250 1)(56 0 0 500 250 1)(57 0 0 500 250 1)(58 0 0 500 250 1)(59 0 0 500 250 1)(60 0 0 500 250 1)(61 0 0 500 250 1)(62 0 0 500 250 1)(63 0 0 500 250 1)(64 0 0 500 250 1)(65 0 0 500 250 1)(66 0 0 500 250 1)(67 0 0 500 250 1)(68 0 0 500 250 1)(69 0 0 500 250 1)(70 0 0 500 250 1)(71 0 0 500 250 1)(72 0 0 500 250 1)(73 0 0 500 250 1)(74 0 0 500 250 1)(75 0 0 500 250 1)(76 0 0 500 250 1)(77 0 0 500 250 1)(78 0 0 500 250 1)(79 0 0 500 250 1)(80 0 0 500 250 1)(81 0 0 500 250 1)(82 0 0 500 250 1)(83 0 0 500 250 1)(84 0 0 500 250 1)(85 0 0 500 250 1)(86 0 0 500 250 1)(87 0 0 500 250 1)(88 0 0 500 250 1)(89 0 0 500 250 1)(90 0 0 500 250 1)(91 0 0 500 250 1)(92 0 0 500 250 1)(93 0 0 500 250 1)(94 0 0 500 250 1)(95 0 0 500 250 1)(96 0 0 500 250 1)(97 0 0 500 250 1)(98 0 0 500 250 1)(99 0 0 500 250 1)(100 0 0 500 250 1)(101 0 0 500 250 1)(102 0 0 500 250 1)(103 0 0 500 250 1)(104 0 0 500 250 1)(105 0 0 500 250 1)(106 0 0 500 250 1)(107 0 0 500 250 1)(108 0 0 500 250 1)(109 0 0 500 250 1)(110 0 0 500 250 1)(111 0 0 500 250 1)(112 0 0 500 250 1)(113 0 0 500 250 1)(114 0 0 500 250 1)(115 0 0 500 250 1)(116 0 0 500 250 1)(117 0 0 500 250 1)(118 0 0 500 250 1)(119 0 0 500 250 1)(120 0 0 500 250 1)(121 0 0 500 250 1)(122 0 0 500 250 1)(123 0 0 500 250 1)(124 0 0 500 250 1)(125 0 0 500 250 1)(126 0 0 500 250 1)(127 0 0 500 250 1)(128 0 0 500 250 1)(129 0 0 500 250 1)(130 0 0 500 250 1)(131 0 0 500 250 1)(132 0 0 500 250 1)(133 0 0 500 250 1)(134 0 0 500 250 1)(135 0 0 500 250 1)(136 0 0 500 250 1)(137 0 0 500 250 1)(138 0 0 500 250 1)(139 0 0 500 250 1)(140 0 0 500 250 1)(141 0 0 500 250 1)(142 0 0 500 250 1)(143 0 0 500 250 1)(144 0 0 500 250 1)(145 0 0 500 250 1)(146 0 0 500 250 1)(147 0 0 500 250 1)(148 0 0 500 250 1)(149 0 0 500 250 1)(150 0 0 500 250 1)(151 0 0 500 250 1)(152 0 0 500 250 1)(153 0 0 500 250 1)(154 0 0 500 250 1)(155 0 0 500 250 1)(156 0 0 500 250 1)(157 0 0 500 250 1)(158 0 0 500 250 1)(159 0 0 500 250 1)(160 0 0 500 250 1)(161 0 0 500 250 1)(162 0 0 500 250 1)(163 0 0 500 250 1)(164 0 0 500 250 1)(165 0 0 500 250 1)(166 0 0 500 250 1)(167 0 0 500 250 1)(168 0 0 500 250 1)(169 0 0 500 250 1)(170 0 0 500 250 1)(171 0 0 500 250 1)(172 0 0 500 250 1)(173 0 0 500 250 1)(174 0 0 500 250 1)(175 0 0 500 250 1)(176 0 0 500 250 1)(177 0 0 500 250 1)(178 0 0 500 250 1)(179 0 0 500 250 1)(180 0 0 500 250 1)(181 0 0 500 250 1)(182 0 0 500 250 1)(183 0 0 500 250 1)(184 0 0 500 250 1)(185 0 0 500 250 1)(186 0 0 500 250 1)(187 0 0 500 250 1)(188 0 0 500 250 1)(189 0 0 500 250 1)(190 0 0 500 250 1)(191 0 0 500 250 1)(192 0 0 500 250 1)(193 0 0 500 250 1)(194 0 0 500 250 1)(195 0 0 500 250 1)(196 0 0 500 250 1)(197 0 0 500 250 1)(198 0 0 500 250 1)(199 0 0 500 250 1)(200 0 0 500 250 1)(201 0 0 500 250 1)(202 0 0 500 250 1)(203 0 0 500 250 1)(204 0 0 500 250 1)(205 0 0 500 250 1)(206 0 0 500 250 1)(207 0 0 500 250 1)(208 0 0 500 250 1)(209 0 0 500 250 1)(210 0 0 500 250 1)(211 0 0 500 250 1)(212 0 0 500 250 1)(213 0 0 500 250 1)(214 0 0 500 250 1)(215 0 0 500 250 1)(216 0 0 500 250 1)(217 0 0 500 250 1)(218 0 0 500 250 1)(219 0 0 500 250 1)(220 0 0 500 250 1)(221 0 0 500 250 1)(222 0 0 500 250 1)(223 0 0 500 250 1)(224 0 0 500 250 1)(225 0 0 500 250 1)(226 0 0 500 250 1)(227 0 0 500 250 1)(228 0 0 500 250 1)(229 0 0 500 250 1)(230 0 0 500 250 1)(231 0 0 500 250 1)(232 0 0 500 250 1)(233 0 0 500 250 1)(234 0 0 500 250 1)(235 0 0 500 250 1)(236 0 0 500 250 1)(237 0 0 500 250 1)(238 0 0 500 250 1)(239 0 0 500 250 1)(240 0 0 500 250 1)(241 0 0 500 250 1)(242 0 0 500 250 1)(243 0 0 500 250 1)(244 0 0 500 250 1)(245 0 0 500 250 1)(246 0 0 500 250 1)(247 0 0 500 250 1)(248 0 0 500 250 1)(249 0 0 500 250 1)(250 0 0 500 250 1)(251 0 0 500 250 1)(252 0 0 500 250 1)(253 0 0 500 250 1)(254 0 0 500 250 1)(255 0 0 500 250 1)(256 0 0 500 250 1)(257 0 0 500 250 1)(258 0 0 500 250 1)(259 0 0 500 250 1)(260 0 0 500 250 1)(261 0 0 500 250 1)(262 0 0 500 250 1)(263 0 0 500 250 1)(264 0 0 500 250 1)(265 0 0 500 250 1)(266 0 0 500 250 1)(267 0 0 500 250 1)(268 0 0 500 250 1)(269 0 0 500 250 1)(270 0 0 500 250 1)(271 0 0 500 250 1)(272 0 0 500 250 1)(273 0 0 500 250 1)(274 0 0 500 250 1)(275 0 0 500 250 1)(276 0 0 500 250 1)(277 0 0 500 250 1)(278 0 0 500 250 1)(279 0 0 500 250 1)(280 0 0 500 250 1)(281 0 0 500 250 1)(282 0 0 500 250 1)(283 0 0 500 250 1)(284 0 0 500 250 1)(285 0 0 500 250 1)(286 0 0 500 250 1)(287 0 0 500 250 1)(288 0 0 500 250 1)(289 0 0 500 250 1)(290 0 0 500 250 1)(291 0 0 500 250 1)(292 0 0 500 250 1)(293 0 0 500 250 1)(294 0 0 500 250 1)(295 0 0 500 250 1)(296 0 0 500 250 1)(297 0 0 500 250 1)(298 0 0 500 250 1)(299 0 0 500 250 1)(300 0 0 500 250 1)(301 0 0 500 250 1)(302 0 0 500 250 1)(303 0 0 500 250 1)(304 0 0 500 250 1)(305 0 0 500 250 1)(306 0 0 500 250 1)(307 0 0 500 250 1)(308 0 0 500 250 1)(309 0 0 500 250 1)(310 0 0 500 250 1)(311 0 0 500 250 1)(312 0 0 500 250 1)(313 0 0 500 250 1)(314 0 0 500 250 1)(315 0 0 500 250 1)(316 0 0 500 250 1)(317 0 0 500 250 1)(318 0 0 500 250 1)(319 0 0 500 250 1)(320 0 0 500 250 1)(321 0 0 500 250 1)(322 0 0 500 250 1)(323 0 0 500 250 1)(324 0 0 500 250 1)(325 0 0 500 250 1)(326 0 0 500 250 1)(327 0 0 500 250 1)(328 0 0 500 250 1)(329 0 0 500 250 1)(330 0 0 500 250 1)(331 0 0 500 250 1)(332 0 0 500 250 1)(333 0 0 500 250 1)(334 0 0 500 250 1)(335 0 0 500 250 1)(336 0 0 500 250 1)(337 0 0 500 250 1)(338 0 0 500 250 1)(339 0 0 500 250 1)(340 0 0 500 250 1)(341 0 0 500 250 1)(342 0 0 500 250 1)(343 0 0 500 250 1)(344 0 0 500 250 1)(345 0 0 500 250 1)(346 0 0 500 250 1)(347 0 0 500 250 1)(348 0 0 500 250 1)(349 0 0 500 250 1)(350 0 0 500 250 1)(351 0 0 500 250 1)(352 0 0 500 250 1)(353 0 0 500 250 1)(354 0 0 500 250 1)(355 0 0 500 250 1)(356 0 0 500 250 1)(357 0 0 500 250 1)(358 0 0 500 250 1)(359 0 0 500 250 1)(360 0 0 500 250 1)(361 0 0 500 250 1)(362 0 0 500 250 1)(363 0 0 500 250 1)(364 0 0 500 250 1)(365 0 0 500 250 1)(366 0 0 500 250 1)(367 0 0 500 250 1)(368 0 0 500 250 1)(369 0 0 500 250 1)(370 0 0 500 250 1)(371 0 0 500 250 1)(372 0 0 500 250 1)(373 0 0 500 250 1)(374 0 0 500 250 1)(375 0 0 500 250 1)(376 0 0 500 250 1)(377 0 0 500 250 1)(378 0 0 500 250 1)(379 0 0 500 250 1)(380 0 0 500 250 1)(381 0 0 500 250 1)(382 0 0 500 250 1)(383 0 0 500 250 1)',
                '~snsChanMap': '(384,384,1)(AP0;0:0)(AP1;1:1)(AP2;2:2)(AP3;3:3)(AP4;4:4)(AP5;5:5)(AP6;6:6)(AP7;7:7)(AP8;8:8)(AP9;9:9)(AP10;10:10)(AP11;11:11)(AP12;12:12)(AP13;13:13)(AP14;14:14)(AP15;15:15)(AP16;16:16)(AP17;17:17)(AP18;18:18)(AP19;19:19)(AP20;20:20)(AP21;21:21)(AP22;22:22)(AP23;23:23)(AP24;24:24)(AP25;25:25)(AP26;26:26)(AP27;27:27)(AP28;28:28)(AP29;29:29)(AP30;30:30)(AP31;31:31)(AP32;32:32)(AP33;33:33)(AP34;34:34)(AP35;35:35)(AP36;36:36)(AP37;37:37)(AP38;38:38)(AP39;39:39)(AP40;40:40)(AP41;41:41)(AP42;42:42)(AP43;43:43)(AP44;44:44)(AP45;45:45)(AP46;46:46)(AP47;47:47)(AP48;48:48)(AP49;49:49)(AP50;50:50)(AP51;51:51)(AP52;52:52)(AP53;53:53)(AP54;54:54)(AP55;55:55)(AP56;56:56)(AP57;57:57)(AP58;58:58)(AP59;59:59)(AP60;60:60)(AP61;61:61)(AP62;62:62)(AP63;63:63)(AP64;64:64)(AP65;65:65)(AP66;66:66)(AP67;67:67)(AP68;68:68)(AP69;69:69)(AP70;70:70)(AP71;71:71)(AP72;72:72)(AP73;73:73)(AP74;74:74)(AP75;75:75)(AP76;76:76)(AP77;77:77)(AP78;78:78)(AP79;79:79)(AP80;80:80)(AP81;81:81)(AP82;82:82)(AP83;83:83)(AP84;84:84)(AP85;85:85)(AP86;86:86)(AP87;87:87)(AP88;88:88)(AP89;89:89)(AP90;90:90)(AP91;91:91)(AP92;92:92)(AP93;93:93)(AP94;94:94)(AP95;95:95)(AP96;96:96)(AP97;97:97)(AP98;98:98)(AP99;99:99)(AP100;100:100)(AP101;101:101)(AP102;102:102)(AP103;103:103)(AP104;104:104)(AP105;105:105)(AP106;106:106)(AP107;107:107)(AP108;108:108)(AP109;109:109)(AP110;110:110)(AP111;111:111)(AP112;112:112)(AP113;113:113)(AP114;114:114)(AP115;115:115)(AP116;116:116)(AP117;117:117)(AP118;118:118)(AP119;119:119)(AP120;120:120)(AP121;121:121)(AP122;122:122)(AP123;123:123)(AP124;124:124)(AP125;125:125)(AP126;126:126)(AP127;127:127)(AP128;128:128)(AP129;129:129)(AP130;130:130)(AP131;131:131)(AP132;132:132)(AP133;133:133)(AP134;134:134)(AP135;135:135)(AP136;136:136)(AP137;137:137)(AP138;138:138)(AP139;139:139)(AP140;140:140)(AP141;141:141)(AP142;142:142)(AP143;143:143)(AP144;144:144)(AP145;145:145)(AP146;146:146)(AP147;147:147)(AP148;148:148)(AP149;149:149)(AP150;150:150)(AP151;151:151)(AP152;152:152)(AP153;153:153)(AP154;154:154)(AP155;155:155)(AP156;156:156)(AP157;157:157)(AP158;158:158)(AP159;159:159)(AP160;160:160)(AP161;161:161)(AP162;162:162)(AP163;163:163)(AP164;164:164)(AP165;165:165)(AP166;166:166)(AP167;167:167)(AP168;168:168)(AP169;169:169)(AP170;170:170)(AP171;171:171)(AP172;172:172)(AP173;173:173)(AP174;174:174)(AP175;175:175)(AP176;176:176)(AP177;177:177)(AP178;178:178)(AP179;179:179)(AP180;180:180)(AP181;181:181)(AP182;182:182)(AP183;183:183)(AP184;184:184)(AP185;185:185)(AP186;186:186)(AP187;187:187)(AP188;188:188)(AP189;189:189)(AP190;190:190)(AP191;191:191)(AP192;192:192)(AP193;193:193)(AP194;194:194)(AP195;195:195)(AP196;196:196)(AP197;197:197)(AP198;198:198)(AP199;199:199)(AP200;200:200)(AP201;201:201)(AP202;202:202)(AP203;203:203)(AP204;204:204)(AP205;205:205)(AP206;206:206)(AP207;207:207)(AP208;208:208)(AP209;209:209)(AP210;210:210)(AP211;211:211)(AP212;212:212)(AP213;213:213)(AP214;214:214)(AP215;215:215)(AP216;216:216)(AP217;217:217)(AP218;218:218)(AP219;219:219)(AP220;220:220)(AP221;221:221)(AP222;222:222)(AP223;223:223)(AP224;224:224)(AP225;225:225)(AP226;226:226)(AP227;227:227)(AP228;228:228)(AP229;229:229)(AP230;230:230)(AP231;231:231)(AP232;232:232)(AP233;233:233)(AP234;234:234)(AP235;235:235)(AP236;236:236)(AP237;237:237)(AP238;238:238)(AP239;239:239)(AP240;240:240)(AP241;241:241)(AP242;242:242)(AP243;243:243)(AP244;244:244)(AP245;245:245)(AP246;246:246)(AP247;247:247)(AP248;248:248)(AP249;249:249)(AP250;250:250)(AP251;251:251)(AP252;252:252)(AP253;253:253)(AP254;254:254)(AP255;255:255)(AP256;256:256)(AP257;257:257)(AP258;258:258)(AP259;259:259)(AP260;260:260)(AP261;261:261)(AP262;262:262)(AP263;263:263)(AP264;264:264)(AP265;265:265)(AP266;266:266)(AP267;267:267)(AP268;268:268)(AP269;269:269)(AP270;270:270)(AP271;271:271)(AP272;272:272)(AP273;273:273)(AP274;274:274)(AP275;275:275)(AP276;276:276)(AP277;277:277)(AP278;278:278)(AP279;279:279)(AP280;280:280)(AP281;281:281)(AP282;282:282)(AP283;283:283)(AP284;284:284)(AP285;285:285)(AP286;286:286)(AP287;287:287)(AP288;288:288)(AP289;289:289)(AP290;290:290)(AP291;291:291)(AP292;292:292)(AP293;293:293)(AP294;294:294)(AP295;295:295)(AP296;296:296)(AP297;297:297)(AP298;298:298)(AP299;299:299)(AP300;300:300)(AP301;301:301)(AP302;302:302)(AP303;303:303)(AP304;304:304)(AP305;305:305)(AP306;306:306)(AP307;307:307)(AP308;308:308)(AP309;309:309)(AP310;310:310)(AP311;311:311)(AP312;312:312)(AP313;313:313)(AP314;314:314)(AP315;315:315)(AP316;316:316)(AP317;317:317)(AP318;318:318)(AP319;319:319)(AP320;320:320)(AP321;321:321)(AP322;322:322)(AP323;323:323)(AP324;324:324)(AP325;325:325)(AP326;326:326)(AP327;327:327)(AP328;328:328)(AP329;329:329)(AP330;330:330)(AP331;331:331)(AP332;332:332)(AP333;333:333)(AP334;334:334)(AP335;335:335)(AP336;336:336)(AP337;337:337)(AP338;338:338)(AP339;339:339)(AP340;340:340)(AP341;341:341)(AP342;342:342)(AP343;343:343)(AP344;344:344)(AP345;345:345)(AP346;346:346)(AP347;347:347)(AP348;348:348)(AP349;349:349)(AP350;350:350)(AP351;351:351)(AP352;352:352)(AP353;353:353)(AP354;354:354)(AP355;355:355)(AP356;356:356)(AP357;357:357)(AP358;358:358)(AP359;359:359)(AP360;360:360)(AP361;361:361)(AP362;362:362)(AP363;363:363)(AP364;364:364)(AP365;365:365)(AP366;366:366)(AP367;367:367)(AP368;368:368)(AP369;369:369)(AP370;370:370)(AP371;371:371)(AP372;372:372)(AP373;373:373)(AP374;374:374)(AP375;375:375)(AP376;376:376)(AP377;377:377)(AP378;378:378)(AP379;379:379)(AP380;380:380)(AP381;381:381)(AP382;382:382)(AP383;383:383)(SY0;768:768)',
                '~snsShankMap': '(1,2,480)(0:0:0:1)(0:1:0:1)(0:0:1:1)(0:1:1:1)(0:0:2:1)(0:1:2:1)(0:0:3:1)(0:1:3:1)(0:0:4:1)(0:1:4:1)(0:0:5:1)(0:1:5:1)(0:0:6:1)(0:1:6:1)(0:0:7:1)(0:1:7:1)(0:0:8:1)(0:1:8:1)(0:0:9:1)(0:1:9:1)(0:0:10:1)(0:1:10:1)(0:0:11:1)(0:1:11:1)(0:0:12:1)(0:1:12:1)(0:0:13:1)(0:1:13:1)(0:0:14:1)(0:1:14:1)(0:0:15:1)(0:1:15:1)(0:0:16:1)(0:1:16:1)(0:0:17:1)(0:1:17:1)(0:0:18:1)(0:1:18:1)(0:0:19:1)(0:1:19:1)(0:0:20:1)(0:1:20:1)(0:0:21:1)(0:1:21:1)(0:0:22:1)(0:1:22:1)(0:0:23:1)(0:1:23:1)(0:0:24:1)(0:1:24:1)(0:0:25:1)(0:1:25:1)(0:0:26:1)(0:1:26:1)(0:0:27:1)(0:1:27:1)(0:0:28:1)(0:1:28:1)(0:0:29:1)(0:1:29:1)(0:0:30:1)(0:1:30:1)(0:0:31:1)(0:1:31:1)(0:0:32:1)(0:1:32:1)(0:0:33:1)(0:1:33:1)(0:0:34:1)(0:1:34:1)(0:0:35:1)(0:1:35:1)(0:0:36:1)(0:1:36:1)(0:0:37:1)(0:1:37:1)(0:0:38:1)(0:1:38:1)(0:0:39:1)(0:1:39:1)(0:0:40:1)(0:1:40:1)(0:0:41:1)(0:1:41:1)(0:0:42:1)(0:1:42:1)(0:0:43:1)(0:1:43:1)(0:0:44:1)(0:1:44:1)(0:0:45:1)(0:1:45:1)(0:0:46:1)(0:1:46:1)(0:0:47:1)(0:1:47:1)(0:0:48:1)(0:1:48:1)(0:0:49:1)(0:1:49:1)(0:0:50:1)(0:1:50:1)(0:0:51:1)(0:1:51:1)(0:0:52:1)(0:1:52:1)(0:0:53:1)(0:1:53:1)(0:0:54:1)(0:1:54:1)(0:0:55:1)(0:1:55:1)(0:0:56:1)(0:1:56:1)(0:0:57:1)(0:1:57:1)(0:0:58:1)(0:1:58:1)(0:0:59:1)(0:1:59:1)(0:0:60:1)(0:1:60:1)(0:0:61:1)(0:1:61:1)(0:0:62:1)(0:1:62:1)(0:0:63:1)(0:1:63:1)(0:0:64:1)(0:1:64:1)(0:0:65:1)(0:1:65:1)(0:0:66:1)(0:1:66:1)(0:0:67:1)(0:1:67:1)(0:0:68:1)(0:1:68:1)(0:0:69:1)(0:1:69:1)(0:0:70:1)(0:1:70:1)(0:0:71:1)(0:1:71:1)(0:0:72:1)(0:1:72:1)(0:0:73:1)(0:1:73:1)(0:0:74:1)(0:1:74:1)(0:0:75:1)(0:1:75:1)(0:0:76:1)(0:1:76:1)(0:0:77:1)(0:1:77:1)(0:0:78:1)(0:1:78:1)(0:0:79:1)(0:1:79:1)(0:0:80:1)(0:1:80:1)(0:0:81:1)(0:1:81:1)(0:0:82:1)(0:1:82:1)(0:0:83:1)(0:1:83:1)(0:0:84:1)(0:1:84:1)(0:0:85:1)(0:1:85:1)(0:0:86:1)(0:1:86:1)(0:0:87:1)(0:1:87:1)(0:0:88:1)(0:1:88:1)(0:0:89:1)(0:1:89:1)(0:0:90:1)(0:1:90:1)(0:0:91:1)(0:1:91:1)(0:0:92:1)(0:1:92:1)(0:0:93:1)(0:1:93:1)(0:0:94:1)(0:1:94:1)(0:0:95:1)(0:1:95:0)(0:0:96:1)(0:1:96:1)(0:0:97:1)(0:1:97:1)(0:0:98:1)(0:1:98:1)(0:0:99:1)(0:1:99:1)(0:0:100:1)(0:1:100:1)(0:0:101:1)(0:1:101:1)(0:0:102:1)(0:1:102:1)(0:0:103:1)(0:1:103:1)(0:0:104:1)(0:1:104:1)(0:0:105:1)(0:1:105:1)(0:0:106:1)(0:1:106:1)(0:0:107:1)(0:1:107:1)(0:0:108:1)(0:1:108:1)(0:0:109:1)(0:1:109:1)(0:0:110:1)(0:1:110:1)(0:0:111:1)(0:1:111:1)(0:0:112:1)(0:1:112:1)(0:0:113:1)(0:1:113:1)(0:0:114:1)(0:1:114:1)(0:0:115:1)(0:1:115:1)(0:0:116:1)(0:1:116:1)(0:0:117:1)(0:1:117:1)(0:0:118:1)(0:1:118:1)(0:0:119:1)(0:1:119:1)(0:0:120:1)(0:1:120:1)(0:0:121:1)(0:1:121:1)(0:0:122:1)(0:1:122:1)(0:0:123:1)(0:1:123:1)(0:0:124:1)(0:1:124:1)(0:0:125:1)(0:1:125:1)(0:0:126:1)(0:1:126:1)(0:0:127:1)(0:1:127:1)(0:0:128:1)(0:1:128:1)(0:0:129:1)(0:1:129:1)(0:0:130:1)(0:1:130:1)(0:0:131:1)(0:1:131:1)(0:0:132:1)(0:1:132:1)(0:0:133:1)(0:1:133:1)(0:0:134:1)(0:1:134:1)(0:0:135:1)(0:1:135:1)(0:0:136:1)(0:1:136:1)(0:0:137:1)(0:1:137:1)(0:0:138:1)(0:1:138:1)(0:0:139:1)(0:1:139:1)(0:0:140:1)(0:1:140:1)(0:0:141:1)(0:1:141:1)(0:0:142:1)(0:1:142:1)(0:0:143:1)(0:1:143:1)(0:0:144:1)(0:1:144:1)(0:0:145:1)(0:1:145:1)(0:0:146:1)(0:1:146:1)(0:0:147:1)(0:1:147:1)(0:0:148:1)(0:1:148:1)(0:0:149:1)(0:1:149:1)(0:0:150:1)(0:1:150:1)(0:0:151:1)(0:1:151:1)(0:0:152:1)(0:1:152:1)(0:0:153:1)(0:1:153:1)(0:0:154:1)(0:1:154:1)(0:0:155:1)(0:1:155:1)(0:0:156:1)(0:1:156:1)(0:0:157:1)(0:1:157:1)(0:0:158:1)(0:1:158:1)(0:0:159:1)(0:1:159:1)(0:0:160:1)(0:1:160:1)(0:0:161:1)(0:1:161:1)(0:0:162:1)(0:1:162:1)(0:0:163:1)(0:1:163:1)(0:0:164:1)(0:1:164:1)(0:0:165:1)(0:1:165:1)(0:0:166:1)(0:1:166:1)(0:0:167:1)(0:1:167:1)(0:0:168:1)(0:1:168:1)(0:0:169:1)(0:1:169:1)(0:0:170:1)(0:1:170:1)(0:0:171:1)(0:1:171:1)(0:0:172:1)(0:1:172:1)(0:0:173:1)(0:1:173:1)(0:0:174:1)(0:1:174:1)(0:0:175:1)(0:1:175:1)(0:0:176:1)(0:1:176:1)(0:0:177:1)(0:1:177:1)(0:0:178:1)(0:1:178:1)(0:0:179:1)(0:1:179:1)(0:0:180:1)(0:1:180:1)(0:0:181:1)(0:1:181:1)(0:0:182:1)(0:1:182:1)(0:0:183:1)(0:1:183:1)(0:0:184:1)(0:1:184:1)(0:0:185:1)(0:1:185:1)(0:0:186:1)(0:1:186:1)(0:0:187:1)(0:1:187:1)(0:0:188:1)(0:1:188:1)(0:0:189:1)(0:1:189:1)(0:0:190:1)(0:1:190:1)(0:0:191:1)(0:1:191:1)',
    }
    with open(metaname, 'w') as f:
        for k in meta_dict.keys():
            f.write(f'{k}={meta_dict[k]}\n')

    np.savez(os.path.splitext(filename)[0] + '_params.npz', 
             st=st_sim, cl=cl_sim, amp=amp_sim, 
             wfid=wfid_sim, drift=drift_sim_ups)

    return st_sim, cl_sim, amp_sim, wf_sim, cb_sim, wfid_sim, drift_sim_ups

# time constants from hybrid
NT = 60000
fs = 30000
nt = 61
twav_min = 20
tw = nt + 60 
ntw = 30 + twav_min
ptw = 30 + nt

def waveforms_from_recording(filename_bg, NT, n_chan_bin, nt, twav_min, chan_map):
    
    # load kilosort4 output
    data_folder = os.path.split(filename_bg)[0]
    ks4_folder = os.path.join(data_folder, 'kilosort4/')   
    ops = np.load(os.path.join(ks4_folder, 'ops.npy'), allow_pickle=True).item()
    st = np.load(os.path.join(ks4_folder, 'spike_times.npy'))
    cl = np.load(os.path.join(ks4_folder, 'spike_clusters.npy'))
    templates = np.load(os.path.join(ks4_folder, 'templates.npy'))
    wf_cb = ((templates**2).sum(axis=-2)**0.5).argmax(axis=1)
    m = pd.read_csv(os.path.join(ks4_folder, 'cluster_KSLabel.tsv'), sep='\t')   
    is_ref = m['KSLabel'].values=="good"
    nc = np.unique(cl, return_counts=True)[1]
    # 0.5 hz firing rate or higher
    cinds = np.nonzero(is_ref & (nc>ops["Nbatches"]))[0]
    
    # remove any spikes before padding window for waveform computation
    cls = cl[st > ntw]
    sts = st[st > ntw]
    iref_c = np.isin(cls, cinds)
    cls = cls[iref_c].astype("int64")
    sts = sts[iref_c].astype("int64")
    n_neurons = len(cinds)
    print('n_neurons = ', len(cinds))
    wf_cb = wf_cb[cinds]

    ncm = len(chan_map)
    wfa = np.zeros((n_neurons, tw, ncm))
    nst = np.zeros(n_neurons, "int")
    tic = time.time()
            
    print('computing waveforms from original recording')
    with io.BinaryRWFile(filename_bg, n_chan_bin, NT=NT) as bfile:
        n_batches = bfile.n_batches
        for ibatch in trange(n_batches):
            bstart = max(0, ibatch * NT - ntw)
            bend = min(bfile.n_samples, (ibatch+1) * NT + ptw)
            slc = slice(bstart, bend)
            data = np.asarray(bfile[slc].T)
            data = data[chan_map]
            ist = np.logical_and(sts >= bstart + ntw, sts < bend - (tw+twav_min))
            j=0
            sts_sub = sts[ist]
            cls_sub = cls[ist]
            for i in cinds:
                ic = cls_sub == i
                if ic.sum() > 0:
                    stb = sts_sub[ic] - bstart - twav_min
                    stb = stb[:,np.newaxis] + np.arange(-ntw+twav_min, ptw)[np.newaxis,:]
                    spks = data[:, stb] 
                    wfa[j] += spks.sum(axis=1).T 
                    nst[j] += ic.sum()
                j+=1
            #if ibatch%100 == 0:
            #    print(f"{ibatch} / {n_batches} batches, time {time.time()-tic:.2f}s")
    
    wfs = np.zeros((n_neurons, tw, n_chan_bin))
    # normalize waveforms by number of spikes
    wfs[:,:,chan_map] = (wfa.copy() / nst[:, np.newaxis, np.newaxis])
    # subtract by baseline per channel
    wfs -= wfs[:,0:10].mean(axis=-2, keepdims=True)
    wfs = wfs.astype("int16")
    # best channel
    wfs_c = chan_map[wf_cb]
    # center waveform on best channel
    xr = wfs_c[:,np.newaxis] + np.arange(-8, 8)
    print(xr.shape)
    xr = xr - np.minimum(0, xr[:,:1])
    xr = xr + np.minimum(0, ncm - 1 - xr[:,-1:])
    # get 1st channel
    chan0 = xr[:,0]
    wfs = wfs[np.arange(0,wfs.shape[0])[:,np.newaxis],:,xr]

    return wfs, cinds, chan0


def generate_hybrid_spikes(filename_bg, chan_map, wfs, cinds, chan0, 
                  n_batches_sim=1350, neuron_seed=11, n_sim=100):
    # firing rates in background dataset
    data_folder = os.path.split(filename_bg)[0]
    ks4_folder = os.path.join(data_folder, 'kilosort4/')   
    templates = np.load(os.path.join(ks4_folder, 'templates.npy'))
    st_bg = np.load(os.path.join(ks4_folder, 'spike_times.npy'))
    clu_bg = np.load(os.path.join(ks4_folder, 'spike_clusters.npy'))
    chan_best = chan_map[((templates**2).sum(axis=-2)**0.5).argmax(axis=1)]
    clu_bg = chan_best[clu_bg].astype("int")

    
    sp_bg = csr_matrix((np.ones_like(clu_bg), (clu_bg, (st_bg/3000).astype("int"))))
    sp_bg = sp_bg.toarray().astype("float32")
    sp_bg = gaussian_filter1d(sp_bg, 10, axis=0)
    sp_bg /= sp_bg.mean(axis=1, keepdims=True)
    plt.imshow(sp_bg, vmin=0, vmax=2, aspect="auto")

    print('creating spike trains')
    # create spikes to add
    np.random.seed(neuron_seed)
    # waveforms
    n_neurons, nch, tw = wfs.shape
    i_sim = np.random.permutation(n_neurons)[:n_sim]
    ci_sim = cinds[i_sim]
    wf_sim = wfs[i_sim].copy()
    cb_sim = chan0[i_sim].copy() 

    # move original waveform up or down 8 channels
    cb_sim += 8 * (2*(np.random.rand(n_sim)>0.5) - 1)
    cb_sim[cb_sim < 0] += 16
    cb_sim[cb_sim > len(chan_map)-17] -= 16

    # firing rates and spike trains
    st_sim = np.zeros(0, 'uint64')
    cl_sim = np.zeros(0, 'uint32')
    frs = np.maximum(0.05, 2.5 * sp_bg[cb_sim])
    t_max = NT * n_batches_sim
    for i in range(n_sim):
        # spike train
        fr = frs[i]
        r = np.random.exponential(1, 10000)
        j = 0
        t = 0
        spi = np.zeros(10000, "uint32")
        while t < t_max:
            isi = int(r[j] * 30000 / fr[int(t//3000)])
            # refractory period
            isi += 60        
            t += isi
            spi[j] = t
            j += 1
        spi = spi[:j]
        st_sim = np.append(st_sim, spi, axis=0)
        cl_sim = np.append(cl_sim, i * np.ones(len(spi), 'uint32'), axis=0)
        
    # re-sort by time
    ii = st_sim.argsort()
    cl_sim = cl_sim[ii]
    st_sim = st_sim[ii]

    # remove spikes after max recording time
    n_max = np.nonzero(st_sim > t_max-ptw-1)[0][0] if st_sim[-1] > t_max-1 else len(st_sim)
    n_min = np.nonzero(st_sim < ntw)[0] 
    n_min = 0 if len(n_min) == 0 else n_min[-1] + 1
    st_sim = st_sim[n_min : n_max]
    cl_sim = cl_sim[n_min : n_max]
    print(len(st_sim))

    # create spikes with waveforms
    n_batches_sim = min(n_batches_sim, int(np.ceil((st_sim.max() + tw) / NT)))
    print(f"n_batches_sim = {n_batches_sim}")
    data = np.zeros((NT * n_batches_sim, 385), "int16")
    n_splits = 3
    tic=time.time()
    # loop over neurons
    print(f'creating spikes from {n_sim} neurons to add')
    for i in trange(n_sim):
        # channels for waveform
        slc_c = slice(cb_sim[i], cb_sim[i]+nch)
        # spikes
        sti = st_sim[cl_sim == i]
        wfi = wf_sim[i].T
        for k in range(n_splits):
            ki = np.arange(k, len(sti), n_splits)
            # indices for spike waveform
            stb = sti[ki] - int(ntw)
            stb = stb[:,np.newaxis].astype(int) + np.arange(0, tw, 1, int)
            data[stb, slc_c] += wfi
               
    print(f'created spikes of size {data.shape}')

    frsim = np.unique(cl_sim, return_counts=True)[1]
    print(f'mean firing rate = {frsim.mean()/(st_sim.max()/fs):.2f}')

    return data, (st_sim, cl_sim, ci_sim, wf_sim, cb_sim)

def hybrid_simulation(filename_bg, sim_name, n_batches_sim=1350, 
                        wfs=None, cinds=None, chan0=None):
    
    n_chan_bin = 385
    probe = io.load_probe("/home/carsen/.kilosort/probes/neuropixPhase3B1_kilosortChanMap.prb")
    chan_map = probe["chanMap"]
    
    filename = f'/media/carsen/ssd1/spikesorting/sim_{sim_name}/sim.imec0.ap.bin'
    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    print("writing to file:")
    print(filename)

    ### compute waveforms in ground truth
    if wfs is None:
        wfs, cinds, chan0 = waveforms_from_recording(filename_bg, NT, n_chan_bin, nt, twav_min, chan_map)

    ### generate spike trains with similar firing profile moved +/- 8 channels
    data, params = generate_hybrid_spikes(filename_bg, chan_map, wfs, cinds, chan0, 
                                             n_batches_sim=n_batches_sim)
    st_sim, cl_sim, ci_sim, wf_sim, cb_sim = params

    print("using original recording as background for new spikes")
    # writing binary
    with io.BinaryRWFile(filename_bg, n_chan_bin, NT=NT) as bfile_bg:
        with io.BinaryRWFile(filename, n_chan_bin, NT=NT, 
                                write=True, n_samples=data.shape[0]) as bfile:
            for ibatch in trange(n_batches_sim):
                slc = slice(ibatch * NT, 
                        min(n_batches_sim * NT, (ibatch+1) * NT))
                bg = np.asarray(bfile_bg[slc]).copy()
                #X = np.zeros((slc.stop - slc.start, n_chan_bin), "float32")
                X = np.asarray(data[slc]).copy()#
                X = X + bg
                to_write = np.clip(X, -2**15 + 1, 2**15 - 1).astype('int16')
                bfile[slc] = to_write
                
    # create meta file for probe
    metaname = os.path.splitext(filename)[0] + '.meta'
    meta_dict = {'fileSizeBytes': os.path.getsize(filename),
                'typeThis': 'imec',
                'imSampRate': 30000,
                'imAiRangeMax': 0.6,
                'imAiRangeMin': -0.6,
                'nSavedChans': 385,
                'snsApLfSy': '384,0,1',
                'snsSaveChanSubset': '0:383,768',
                '~imroTbl': '(0,384)(0 0 0 500 250 1)(1 0 0 500 250 1)(2 0 0 500 250 1)(3 0 0 500 250 1)(4 0 0 500 250 1)(5 0 0 500 250 1)(6 0 0 500 250 1)(7 0 0 500 250 1)(8 0 0 500 250 1)(9 0 0 500 250 1)(10 0 0 500 250 1)(11 0 0 500 250 1)(12 0 0 500 250 1)(13 0 0 500 250 1)(14 0 0 500 250 1)(15 0 0 500 250 1)(16 0 0 500 250 1)(17 0 0 500 250 1)(18 0 0 500 250 1)(19 0 0 500 250 1)(20 0 0 500 250 1)(21 0 0 500 250 1)(22 0 0 500 250 1)(23 0 0 500 250 1)(24 0 0 500 250 1)(25 0 0 500 250 1)(26 0 0 500 250 1)(27 0 0 500 250 1)(28 0 0 500 250 1)(29 0 0 500 250 1)(30 0 0 500 250 1)(31 0 0 500 250 1)(32 0 0 500 250 1)(33 0 0 500 250 1)(34 0 0 500 250 1)(35 0 0 500 250 1)(36 0 0 500 250 1)(37 0 0 500 250 1)(38 0 0 500 250 1)(39 0 0 500 250 1)(40 0 0 500 250 1)(41 0 0 500 250 1)(42 0 0 500 250 1)(43 0 0 500 250 1)(44 0 0 500 250 1)(45 0 0 500 250 1)(46 0 0 500 250 1)(47 0 0 500 250 1)(48 0 0 500 250 1)(49 0 0 500 250 1)(50 0 0 500 250 1)(51 0 0 500 250 1)(52 0 0 500 250 1)(53 0 0 500 250 1)(54 0 0 500 250 1)(55 0 0 500 250 1)(56 0 0 500 250 1)(57 0 0 500 250 1)(58 0 0 500 250 1)(59 0 0 500 250 1)(60 0 0 500 250 1)(61 0 0 500 250 1)(62 0 0 500 250 1)(63 0 0 500 250 1)(64 0 0 500 250 1)(65 0 0 500 250 1)(66 0 0 500 250 1)(67 0 0 500 250 1)(68 0 0 500 250 1)(69 0 0 500 250 1)(70 0 0 500 250 1)(71 0 0 500 250 1)(72 0 0 500 250 1)(73 0 0 500 250 1)(74 0 0 500 250 1)(75 0 0 500 250 1)(76 0 0 500 250 1)(77 0 0 500 250 1)(78 0 0 500 250 1)(79 0 0 500 250 1)(80 0 0 500 250 1)(81 0 0 500 250 1)(82 0 0 500 250 1)(83 0 0 500 250 1)(84 0 0 500 250 1)(85 0 0 500 250 1)(86 0 0 500 250 1)(87 0 0 500 250 1)(88 0 0 500 250 1)(89 0 0 500 250 1)(90 0 0 500 250 1)(91 0 0 500 250 1)(92 0 0 500 250 1)(93 0 0 500 250 1)(94 0 0 500 250 1)(95 0 0 500 250 1)(96 0 0 500 250 1)(97 0 0 500 250 1)(98 0 0 500 250 1)(99 0 0 500 250 1)(100 0 0 500 250 1)(101 0 0 500 250 1)(102 0 0 500 250 1)(103 0 0 500 250 1)(104 0 0 500 250 1)(105 0 0 500 250 1)(106 0 0 500 250 1)(107 0 0 500 250 1)(108 0 0 500 250 1)(109 0 0 500 250 1)(110 0 0 500 250 1)(111 0 0 500 250 1)(112 0 0 500 250 1)(113 0 0 500 250 1)(114 0 0 500 250 1)(115 0 0 500 250 1)(116 0 0 500 250 1)(117 0 0 500 250 1)(118 0 0 500 250 1)(119 0 0 500 250 1)(120 0 0 500 250 1)(121 0 0 500 250 1)(122 0 0 500 250 1)(123 0 0 500 250 1)(124 0 0 500 250 1)(125 0 0 500 250 1)(126 0 0 500 250 1)(127 0 0 500 250 1)(128 0 0 500 250 1)(129 0 0 500 250 1)(130 0 0 500 250 1)(131 0 0 500 250 1)(132 0 0 500 250 1)(133 0 0 500 250 1)(134 0 0 500 250 1)(135 0 0 500 250 1)(136 0 0 500 250 1)(137 0 0 500 250 1)(138 0 0 500 250 1)(139 0 0 500 250 1)(140 0 0 500 250 1)(141 0 0 500 250 1)(142 0 0 500 250 1)(143 0 0 500 250 1)(144 0 0 500 250 1)(145 0 0 500 250 1)(146 0 0 500 250 1)(147 0 0 500 250 1)(148 0 0 500 250 1)(149 0 0 500 250 1)(150 0 0 500 250 1)(151 0 0 500 250 1)(152 0 0 500 250 1)(153 0 0 500 250 1)(154 0 0 500 250 1)(155 0 0 500 250 1)(156 0 0 500 250 1)(157 0 0 500 250 1)(158 0 0 500 250 1)(159 0 0 500 250 1)(160 0 0 500 250 1)(161 0 0 500 250 1)(162 0 0 500 250 1)(163 0 0 500 250 1)(164 0 0 500 250 1)(165 0 0 500 250 1)(166 0 0 500 250 1)(167 0 0 500 250 1)(168 0 0 500 250 1)(169 0 0 500 250 1)(170 0 0 500 250 1)(171 0 0 500 250 1)(172 0 0 500 250 1)(173 0 0 500 250 1)(174 0 0 500 250 1)(175 0 0 500 250 1)(176 0 0 500 250 1)(177 0 0 500 250 1)(178 0 0 500 250 1)(179 0 0 500 250 1)(180 0 0 500 250 1)(181 0 0 500 250 1)(182 0 0 500 250 1)(183 0 0 500 250 1)(184 0 0 500 250 1)(185 0 0 500 250 1)(186 0 0 500 250 1)(187 0 0 500 250 1)(188 0 0 500 250 1)(189 0 0 500 250 1)(190 0 0 500 250 1)(191 0 0 500 250 1)(192 0 0 500 250 1)(193 0 0 500 250 1)(194 0 0 500 250 1)(195 0 0 500 250 1)(196 0 0 500 250 1)(197 0 0 500 250 1)(198 0 0 500 250 1)(199 0 0 500 250 1)(200 0 0 500 250 1)(201 0 0 500 250 1)(202 0 0 500 250 1)(203 0 0 500 250 1)(204 0 0 500 250 1)(205 0 0 500 250 1)(206 0 0 500 250 1)(207 0 0 500 250 1)(208 0 0 500 250 1)(209 0 0 500 250 1)(210 0 0 500 250 1)(211 0 0 500 250 1)(212 0 0 500 250 1)(213 0 0 500 250 1)(214 0 0 500 250 1)(215 0 0 500 250 1)(216 0 0 500 250 1)(217 0 0 500 250 1)(218 0 0 500 250 1)(219 0 0 500 250 1)(220 0 0 500 250 1)(221 0 0 500 250 1)(222 0 0 500 250 1)(223 0 0 500 250 1)(224 0 0 500 250 1)(225 0 0 500 250 1)(226 0 0 500 250 1)(227 0 0 500 250 1)(228 0 0 500 250 1)(229 0 0 500 250 1)(230 0 0 500 250 1)(231 0 0 500 250 1)(232 0 0 500 250 1)(233 0 0 500 250 1)(234 0 0 500 250 1)(235 0 0 500 250 1)(236 0 0 500 250 1)(237 0 0 500 250 1)(238 0 0 500 250 1)(239 0 0 500 250 1)(240 0 0 500 250 1)(241 0 0 500 250 1)(242 0 0 500 250 1)(243 0 0 500 250 1)(244 0 0 500 250 1)(245 0 0 500 250 1)(246 0 0 500 250 1)(247 0 0 500 250 1)(248 0 0 500 250 1)(249 0 0 500 250 1)(250 0 0 500 250 1)(251 0 0 500 250 1)(252 0 0 500 250 1)(253 0 0 500 250 1)(254 0 0 500 250 1)(255 0 0 500 250 1)(256 0 0 500 250 1)(257 0 0 500 250 1)(258 0 0 500 250 1)(259 0 0 500 250 1)(260 0 0 500 250 1)(261 0 0 500 250 1)(262 0 0 500 250 1)(263 0 0 500 250 1)(264 0 0 500 250 1)(265 0 0 500 250 1)(266 0 0 500 250 1)(267 0 0 500 250 1)(268 0 0 500 250 1)(269 0 0 500 250 1)(270 0 0 500 250 1)(271 0 0 500 250 1)(272 0 0 500 250 1)(273 0 0 500 250 1)(274 0 0 500 250 1)(275 0 0 500 250 1)(276 0 0 500 250 1)(277 0 0 500 250 1)(278 0 0 500 250 1)(279 0 0 500 250 1)(280 0 0 500 250 1)(281 0 0 500 250 1)(282 0 0 500 250 1)(283 0 0 500 250 1)(284 0 0 500 250 1)(285 0 0 500 250 1)(286 0 0 500 250 1)(287 0 0 500 250 1)(288 0 0 500 250 1)(289 0 0 500 250 1)(290 0 0 500 250 1)(291 0 0 500 250 1)(292 0 0 500 250 1)(293 0 0 500 250 1)(294 0 0 500 250 1)(295 0 0 500 250 1)(296 0 0 500 250 1)(297 0 0 500 250 1)(298 0 0 500 250 1)(299 0 0 500 250 1)(300 0 0 500 250 1)(301 0 0 500 250 1)(302 0 0 500 250 1)(303 0 0 500 250 1)(304 0 0 500 250 1)(305 0 0 500 250 1)(306 0 0 500 250 1)(307 0 0 500 250 1)(308 0 0 500 250 1)(309 0 0 500 250 1)(310 0 0 500 250 1)(311 0 0 500 250 1)(312 0 0 500 250 1)(313 0 0 500 250 1)(314 0 0 500 250 1)(315 0 0 500 250 1)(316 0 0 500 250 1)(317 0 0 500 250 1)(318 0 0 500 250 1)(319 0 0 500 250 1)(320 0 0 500 250 1)(321 0 0 500 250 1)(322 0 0 500 250 1)(323 0 0 500 250 1)(324 0 0 500 250 1)(325 0 0 500 250 1)(326 0 0 500 250 1)(327 0 0 500 250 1)(328 0 0 500 250 1)(329 0 0 500 250 1)(330 0 0 500 250 1)(331 0 0 500 250 1)(332 0 0 500 250 1)(333 0 0 500 250 1)(334 0 0 500 250 1)(335 0 0 500 250 1)(336 0 0 500 250 1)(337 0 0 500 250 1)(338 0 0 500 250 1)(339 0 0 500 250 1)(340 0 0 500 250 1)(341 0 0 500 250 1)(342 0 0 500 250 1)(343 0 0 500 250 1)(344 0 0 500 250 1)(345 0 0 500 250 1)(346 0 0 500 250 1)(347 0 0 500 250 1)(348 0 0 500 250 1)(349 0 0 500 250 1)(350 0 0 500 250 1)(351 0 0 500 250 1)(352 0 0 500 250 1)(353 0 0 500 250 1)(354 0 0 500 250 1)(355 0 0 500 250 1)(356 0 0 500 250 1)(357 0 0 500 250 1)(358 0 0 500 250 1)(359 0 0 500 250 1)(360 0 0 500 250 1)(361 0 0 500 250 1)(362 0 0 500 250 1)(363 0 0 500 250 1)(364 0 0 500 250 1)(365 0 0 500 250 1)(366 0 0 500 250 1)(367 0 0 500 250 1)(368 0 0 500 250 1)(369 0 0 500 250 1)(370 0 0 500 250 1)(371 0 0 500 250 1)(372 0 0 500 250 1)(373 0 0 500 250 1)(374 0 0 500 250 1)(375 0 0 500 250 1)(376 0 0 500 250 1)(377 0 0 500 250 1)(378 0 0 500 250 1)(379 0 0 500 250 1)(380 0 0 500 250 1)(381 0 0 500 250 1)(382 0 0 500 250 1)(383 0 0 500 250 1)',
                '~snsChanMap': '(384,384,1)(AP0;0:0)(AP1;1:1)(AP2;2:2)(AP3;3:3)(AP4;4:4)(AP5;5:5)(AP6;6:6)(AP7;7:7)(AP8;8:8)(AP9;9:9)(AP10;10:10)(AP11;11:11)(AP12;12:12)(AP13;13:13)(AP14;14:14)(AP15;15:15)(AP16;16:16)(AP17;17:17)(AP18;18:18)(AP19;19:19)(AP20;20:20)(AP21;21:21)(AP22;22:22)(AP23;23:23)(AP24;24:24)(AP25;25:25)(AP26;26:26)(AP27;27:27)(AP28;28:28)(AP29;29:29)(AP30;30:30)(AP31;31:31)(AP32;32:32)(AP33;33:33)(AP34;34:34)(AP35;35:35)(AP36;36:36)(AP37;37:37)(AP38;38:38)(AP39;39:39)(AP40;40:40)(AP41;41:41)(AP42;42:42)(AP43;43:43)(AP44;44:44)(AP45;45:45)(AP46;46:46)(AP47;47:47)(AP48;48:48)(AP49;49:49)(AP50;50:50)(AP51;51:51)(AP52;52:52)(AP53;53:53)(AP54;54:54)(AP55;55:55)(AP56;56:56)(AP57;57:57)(AP58;58:58)(AP59;59:59)(AP60;60:60)(AP61;61:61)(AP62;62:62)(AP63;63:63)(AP64;64:64)(AP65;65:65)(AP66;66:66)(AP67;67:67)(AP68;68:68)(AP69;69:69)(AP70;70:70)(AP71;71:71)(AP72;72:72)(AP73;73:73)(AP74;74:74)(AP75;75:75)(AP76;76:76)(AP77;77:77)(AP78;78:78)(AP79;79:79)(AP80;80:80)(AP81;81:81)(AP82;82:82)(AP83;83:83)(AP84;84:84)(AP85;85:85)(AP86;86:86)(AP87;87:87)(AP88;88:88)(AP89;89:89)(AP90;90:90)(AP91;91:91)(AP92;92:92)(AP93;93:93)(AP94;94:94)(AP95;95:95)(AP96;96:96)(AP97;97:97)(AP98;98:98)(AP99;99:99)(AP100;100:100)(AP101;101:101)(AP102;102:102)(AP103;103:103)(AP104;104:104)(AP105;105:105)(AP106;106:106)(AP107;107:107)(AP108;108:108)(AP109;109:109)(AP110;110:110)(AP111;111:111)(AP112;112:112)(AP113;113:113)(AP114;114:114)(AP115;115:115)(AP116;116:116)(AP117;117:117)(AP118;118:118)(AP119;119:119)(AP120;120:120)(AP121;121:121)(AP122;122:122)(AP123;123:123)(AP124;124:124)(AP125;125:125)(AP126;126:126)(AP127;127:127)(AP128;128:128)(AP129;129:129)(AP130;130:130)(AP131;131:131)(AP132;132:132)(AP133;133:133)(AP134;134:134)(AP135;135:135)(AP136;136:136)(AP137;137:137)(AP138;138:138)(AP139;139:139)(AP140;140:140)(AP141;141:141)(AP142;142:142)(AP143;143:143)(AP144;144:144)(AP145;145:145)(AP146;146:146)(AP147;147:147)(AP148;148:148)(AP149;149:149)(AP150;150:150)(AP151;151:151)(AP152;152:152)(AP153;153:153)(AP154;154:154)(AP155;155:155)(AP156;156:156)(AP157;157:157)(AP158;158:158)(AP159;159:159)(AP160;160:160)(AP161;161:161)(AP162;162:162)(AP163;163:163)(AP164;164:164)(AP165;165:165)(AP166;166:166)(AP167;167:167)(AP168;168:168)(AP169;169:169)(AP170;170:170)(AP171;171:171)(AP172;172:172)(AP173;173:173)(AP174;174:174)(AP175;175:175)(AP176;176:176)(AP177;177:177)(AP178;178:178)(AP179;179:179)(AP180;180:180)(AP181;181:181)(AP182;182:182)(AP183;183:183)(AP184;184:184)(AP185;185:185)(AP186;186:186)(AP187;187:187)(AP188;188:188)(AP189;189:189)(AP190;190:190)(AP191;191:191)(AP192;192:192)(AP193;193:193)(AP194;194:194)(AP195;195:195)(AP196;196:196)(AP197;197:197)(AP198;198:198)(AP199;199:199)(AP200;200:200)(AP201;201:201)(AP202;202:202)(AP203;203:203)(AP204;204:204)(AP205;205:205)(AP206;206:206)(AP207;207:207)(AP208;208:208)(AP209;209:209)(AP210;210:210)(AP211;211:211)(AP212;212:212)(AP213;213:213)(AP214;214:214)(AP215;215:215)(AP216;216:216)(AP217;217:217)(AP218;218:218)(AP219;219:219)(AP220;220:220)(AP221;221:221)(AP222;222:222)(AP223;223:223)(AP224;224:224)(AP225;225:225)(AP226;226:226)(AP227;227:227)(AP228;228:228)(AP229;229:229)(AP230;230:230)(AP231;231:231)(AP232;232:232)(AP233;233:233)(AP234;234:234)(AP235;235:235)(AP236;236:236)(AP237;237:237)(AP238;238:238)(AP239;239:239)(AP240;240:240)(AP241;241:241)(AP242;242:242)(AP243;243:243)(AP244;244:244)(AP245;245:245)(AP246;246:246)(AP247;247:247)(AP248;248:248)(AP249;249:249)(AP250;250:250)(AP251;251:251)(AP252;252:252)(AP253;253:253)(AP254;254:254)(AP255;255:255)(AP256;256:256)(AP257;257:257)(AP258;258:258)(AP259;259:259)(AP260;260:260)(AP261;261:261)(AP262;262:262)(AP263;263:263)(AP264;264:264)(AP265;265:265)(AP266;266:266)(AP267;267:267)(AP268;268:268)(AP269;269:269)(AP270;270:270)(AP271;271:271)(AP272;272:272)(AP273;273:273)(AP274;274:274)(AP275;275:275)(AP276;276:276)(AP277;277:277)(AP278;278:278)(AP279;279:279)(AP280;280:280)(AP281;281:281)(AP282;282:282)(AP283;283:283)(AP284;284:284)(AP285;285:285)(AP286;286:286)(AP287;287:287)(AP288;288:288)(AP289;289:289)(AP290;290:290)(AP291;291:291)(AP292;292:292)(AP293;293:293)(AP294;294:294)(AP295;295:295)(AP296;296:296)(AP297;297:297)(AP298;298:298)(AP299;299:299)(AP300;300:300)(AP301;301:301)(AP302;302:302)(AP303;303:303)(AP304;304:304)(AP305;305:305)(AP306;306:306)(AP307;307:307)(AP308;308:308)(AP309;309:309)(AP310;310:310)(AP311;311:311)(AP312;312:312)(AP313;313:313)(AP314;314:314)(AP315;315:315)(AP316;316:316)(AP317;317:317)(AP318;318:318)(AP319;319:319)(AP320;320:320)(AP321;321:321)(AP322;322:322)(AP323;323:323)(AP324;324:324)(AP325;325:325)(AP326;326:326)(AP327;327:327)(AP328;328:328)(AP329;329:329)(AP330;330:330)(AP331;331:331)(AP332;332:332)(AP333;333:333)(AP334;334:334)(AP335;335:335)(AP336;336:336)(AP337;337:337)(AP338;338:338)(AP339;339:339)(AP340;340:340)(AP341;341:341)(AP342;342:342)(AP343;343:343)(AP344;344:344)(AP345;345:345)(AP346;346:346)(AP347;347:347)(AP348;348:348)(AP349;349:349)(AP350;350:350)(AP351;351:351)(AP352;352:352)(AP353;353:353)(AP354;354:354)(AP355;355:355)(AP356;356:356)(AP357;357:357)(AP358;358:358)(AP359;359:359)(AP360;360:360)(AP361;361:361)(AP362;362:362)(AP363;363:363)(AP364;364:364)(AP365;365:365)(AP366;366:366)(AP367;367:367)(AP368;368:368)(AP369;369:369)(AP370;370:370)(AP371;371:371)(AP372;372:372)(AP373;373:373)(AP374;374:374)(AP375;375:375)(AP376;376:376)(AP377;377:377)(AP378;378:378)(AP379;379:379)(AP380;380:380)(AP381;381:381)(AP382;382:382)(AP383;383:383)(SY0;768:768)',
                '~snsShankMap': '(1,2,480)(0:0:0:1)(0:1:0:1)(0:0:1:1)(0:1:1:1)(0:0:2:1)(0:1:2:1)(0:0:3:1)(0:1:3:1)(0:0:4:1)(0:1:4:1)(0:0:5:1)(0:1:5:1)(0:0:6:1)(0:1:6:1)(0:0:7:1)(0:1:7:1)(0:0:8:1)(0:1:8:1)(0:0:9:1)(0:1:9:1)(0:0:10:1)(0:1:10:1)(0:0:11:1)(0:1:11:1)(0:0:12:1)(0:1:12:1)(0:0:13:1)(0:1:13:1)(0:0:14:1)(0:1:14:1)(0:0:15:1)(0:1:15:1)(0:0:16:1)(0:1:16:1)(0:0:17:1)(0:1:17:1)(0:0:18:1)(0:1:18:1)(0:0:19:1)(0:1:19:1)(0:0:20:1)(0:1:20:1)(0:0:21:1)(0:1:21:1)(0:0:22:1)(0:1:22:1)(0:0:23:1)(0:1:23:1)(0:0:24:1)(0:1:24:1)(0:0:25:1)(0:1:25:1)(0:0:26:1)(0:1:26:1)(0:0:27:1)(0:1:27:1)(0:0:28:1)(0:1:28:1)(0:0:29:1)(0:1:29:1)(0:0:30:1)(0:1:30:1)(0:0:31:1)(0:1:31:1)(0:0:32:1)(0:1:32:1)(0:0:33:1)(0:1:33:1)(0:0:34:1)(0:1:34:1)(0:0:35:1)(0:1:35:1)(0:0:36:1)(0:1:36:1)(0:0:37:1)(0:1:37:1)(0:0:38:1)(0:1:38:1)(0:0:39:1)(0:1:39:1)(0:0:40:1)(0:1:40:1)(0:0:41:1)(0:1:41:1)(0:0:42:1)(0:1:42:1)(0:0:43:1)(0:1:43:1)(0:0:44:1)(0:1:44:1)(0:0:45:1)(0:1:45:1)(0:0:46:1)(0:1:46:1)(0:0:47:1)(0:1:47:1)(0:0:48:1)(0:1:48:1)(0:0:49:1)(0:1:49:1)(0:0:50:1)(0:1:50:1)(0:0:51:1)(0:1:51:1)(0:0:52:1)(0:1:52:1)(0:0:53:1)(0:1:53:1)(0:0:54:1)(0:1:54:1)(0:0:55:1)(0:1:55:1)(0:0:56:1)(0:1:56:1)(0:0:57:1)(0:1:57:1)(0:0:58:1)(0:1:58:1)(0:0:59:1)(0:1:59:1)(0:0:60:1)(0:1:60:1)(0:0:61:1)(0:1:61:1)(0:0:62:1)(0:1:62:1)(0:0:63:1)(0:1:63:1)(0:0:64:1)(0:1:64:1)(0:0:65:1)(0:1:65:1)(0:0:66:1)(0:1:66:1)(0:0:67:1)(0:1:67:1)(0:0:68:1)(0:1:68:1)(0:0:69:1)(0:1:69:1)(0:0:70:1)(0:1:70:1)(0:0:71:1)(0:1:71:1)(0:0:72:1)(0:1:72:1)(0:0:73:1)(0:1:73:1)(0:0:74:1)(0:1:74:1)(0:0:75:1)(0:1:75:1)(0:0:76:1)(0:1:76:1)(0:0:77:1)(0:1:77:1)(0:0:78:1)(0:1:78:1)(0:0:79:1)(0:1:79:1)(0:0:80:1)(0:1:80:1)(0:0:81:1)(0:1:81:1)(0:0:82:1)(0:1:82:1)(0:0:83:1)(0:1:83:1)(0:0:84:1)(0:1:84:1)(0:0:85:1)(0:1:85:1)(0:0:86:1)(0:1:86:1)(0:0:87:1)(0:1:87:1)(0:0:88:1)(0:1:88:1)(0:0:89:1)(0:1:89:1)(0:0:90:1)(0:1:90:1)(0:0:91:1)(0:1:91:1)(0:0:92:1)(0:1:92:1)(0:0:93:1)(0:1:93:1)(0:0:94:1)(0:1:94:1)(0:0:95:1)(0:1:95:0)(0:0:96:1)(0:1:96:1)(0:0:97:1)(0:1:97:1)(0:0:98:1)(0:1:98:1)(0:0:99:1)(0:1:99:1)(0:0:100:1)(0:1:100:1)(0:0:101:1)(0:1:101:1)(0:0:102:1)(0:1:102:1)(0:0:103:1)(0:1:103:1)(0:0:104:1)(0:1:104:1)(0:0:105:1)(0:1:105:1)(0:0:106:1)(0:1:106:1)(0:0:107:1)(0:1:107:1)(0:0:108:1)(0:1:108:1)(0:0:109:1)(0:1:109:1)(0:0:110:1)(0:1:110:1)(0:0:111:1)(0:1:111:1)(0:0:112:1)(0:1:112:1)(0:0:113:1)(0:1:113:1)(0:0:114:1)(0:1:114:1)(0:0:115:1)(0:1:115:1)(0:0:116:1)(0:1:116:1)(0:0:117:1)(0:1:117:1)(0:0:118:1)(0:1:118:1)(0:0:119:1)(0:1:119:1)(0:0:120:1)(0:1:120:1)(0:0:121:1)(0:1:121:1)(0:0:122:1)(0:1:122:1)(0:0:123:1)(0:1:123:1)(0:0:124:1)(0:1:124:1)(0:0:125:1)(0:1:125:1)(0:0:126:1)(0:1:126:1)(0:0:127:1)(0:1:127:1)(0:0:128:1)(0:1:128:1)(0:0:129:1)(0:1:129:1)(0:0:130:1)(0:1:130:1)(0:0:131:1)(0:1:131:1)(0:0:132:1)(0:1:132:1)(0:0:133:1)(0:1:133:1)(0:0:134:1)(0:1:134:1)(0:0:135:1)(0:1:135:1)(0:0:136:1)(0:1:136:1)(0:0:137:1)(0:1:137:1)(0:0:138:1)(0:1:138:1)(0:0:139:1)(0:1:139:1)(0:0:140:1)(0:1:140:1)(0:0:141:1)(0:1:141:1)(0:0:142:1)(0:1:142:1)(0:0:143:1)(0:1:143:1)(0:0:144:1)(0:1:144:1)(0:0:145:1)(0:1:145:1)(0:0:146:1)(0:1:146:1)(0:0:147:1)(0:1:147:1)(0:0:148:1)(0:1:148:1)(0:0:149:1)(0:1:149:1)(0:0:150:1)(0:1:150:1)(0:0:151:1)(0:1:151:1)(0:0:152:1)(0:1:152:1)(0:0:153:1)(0:1:153:1)(0:0:154:1)(0:1:154:1)(0:0:155:1)(0:1:155:1)(0:0:156:1)(0:1:156:1)(0:0:157:1)(0:1:157:1)(0:0:158:1)(0:1:158:1)(0:0:159:1)(0:1:159:1)(0:0:160:1)(0:1:160:1)(0:0:161:1)(0:1:161:1)(0:0:162:1)(0:1:162:1)(0:0:163:1)(0:1:163:1)(0:0:164:1)(0:1:164:1)(0:0:165:1)(0:1:165:1)(0:0:166:1)(0:1:166:1)(0:0:167:1)(0:1:167:1)(0:0:168:1)(0:1:168:1)(0:0:169:1)(0:1:169:1)(0:0:170:1)(0:1:170:1)(0:0:171:1)(0:1:171:1)(0:0:172:1)(0:1:172:1)(0:0:173:1)(0:1:173:1)(0:0:174:1)(0:1:174:1)(0:0:175:1)(0:1:175:1)(0:0:176:1)(0:1:176:1)(0:0:177:1)(0:1:177:1)(0:0:178:1)(0:1:178:1)(0:0:179:1)(0:1:179:1)(0:0:180:1)(0:1:180:1)(0:0:181:1)(0:1:181:1)(0:0:182:1)(0:1:182:1)(0:0:183:1)(0:1:183:1)(0:0:184:1)(0:1:184:1)(0:0:185:1)(0:1:185:1)(0:0:186:1)(0:1:186:1)(0:0:187:1)(0:1:187:1)(0:0:188:1)(0:1:188:1)(0:0:189:1)(0:1:189:1)(0:0:190:1)(0:1:190:1)(0:0:191:1)(0:1:191:1)',
    }
    with open(metaname, 'w') as f:
        for k in meta_dict.keys():
            f.write(f'{k}={meta_dict[k]}\n')

    np.savez(os.path.splitext(filename)[0] + '_params.npz', 
                st=st_sim, cl=cl_sim, ci=ci_sim, 
                wfs=wf_sim, cb=cb_sim+8)

    print('hybrid ground truth completed')

    return data, params

