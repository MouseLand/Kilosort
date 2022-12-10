import numpy as np 
import os, time 
import matplotlib.pyplot as plt 
import torch 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from kilosort.datashift import kernel2D
from kilosort import io

def preprocess_wfs_sts(filename, y_chan, x_chan, ups=10, device=torch.device('cuda')):
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
    lam = 1
    wfs_smoothed = np.zeros((wfs.shape[0], wfs.shape[1]*ups, *wfs.shape[2:]),'float32')
    for ic in range(4):
        iwc = wfs_x == ic
        nd,nt,nc = wfs.shape[-3:]
        xp = np.vstack((y_chan[ic : ic+nc], x_chan[ic : ic+nc])).T
        zup = np.arange(0,nd*2,2./ups)[:,np.newaxis,np.newaxis]
        zup = np.concatenate((zup, np.zeros((nd*ups,1,1))), axis=-1)
        xup = (np.tile(xp[np.newaxis,:], (nd*ups,1,1)) + zup).reshape(-1, 2)

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

    return wfs_smoothed, wfs_x, contaminations, sts, cls



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
                    drift_range=5, drift_seed=0, ups=10):
    """ simulate spikes using 
        - random waveforms from wfs with x-pos wfs_x 
        - spike train stats from (st, cl) 
        - drift with smoothing constant of tsig across batches 
        - first n_sim neurons are good neurons, next 2 * n_sim neurons are MUA / noise
    WARNING: requires RAM for full simulation as written

    """
    n_bins, n_twav, nc = wfs.shape[1:] 
    st_sim = np.zeros(0, 'uint64')
    cl_sim = np.zeros(0, 'uint32')
    wf_sim = np.zeros((n_sim + n_noise, n_bins, n_twav, nc), 'float32')
    cb_sim = np.zeros(n_sim + n_noise, 'int')

    # random amplitudes
    # for good neurons
    amp_sim = np.random.exponential(scale=1., size=(n_sim,)) * 12. + 10.
    # for MUA / noise
    amp_sim = np.append(amp_sim, 
                        np.random.rand(n_noise) * 8. + 4.,
                        axis=0)

    # create good neurons
    n_time = st.max() / 30000.

    n_time_sim = n_batches * batch_size
    n_max = np.nonzero(st > n_time_sim-1)[0][0]
    good_neurons = np.nonzero(contaminations < 0.1)[0]
    for i in range(n_sim):
        # random spike train, waveform and channel per neuron
        sp_rand = np.random.randint(cl.max()+1)
        wf_rand = np.random.randint(len(good_neurons))
        cb_rand = np.random.randint(96-1)
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
        else:
            i -= 1
    print('created good neurons')

    # create MUA / noise
    bad_neurons = np.nonzero(contaminations >= 0.1)[0]
    _, frs = np.unique(cl, return_counts=True)
    frs = frs.astype('float32')
    frs /= n_time
    for i in range(n_noise):
        # random firing rate, waveform and channel per neuron
        fr_rand = np.random.randint(len(frs))
        wf_rand = np.random.randint(len(bad_neurons))
        cb_rand = np.random.randint(96-1)
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
    print('created MUA / noise')
        
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
        rand = True
        if rand:
            drift_sim = np.random.randn(n_batches)
            drift_sim = gaussian_filter1d(drift_sim, tsig)
            drift_sim -= drift_sim.min()
            drift_sim /= drift_sim.max()
            drift_sim *= drift_range

            # drift across probe (9 positions)
            drift_chan = np.random.randn(n_batches, 9)
            drift_chan = gaussian_filter1d(drift_chan, tsig, axis=0)
            drift_chan -= drift_chan.min(axis=0) 
            drift_chan /= drift_chan.max(axis=0)
            drift_chan *= 3
            drift_chan = gaussian_filter1d(drift_chan*4, 2, axis=-1)
            drift_chan += drift_sim[:,np.newaxis]
            drift_chan -= drift_chan.min() 
            drift_chan /= drift_chan.max()
            drift_chan *= drift_range
            drift_chan += 10 - drift_range/2
        else:
            drift_chan = 5 + np.tile(np.linspace(0, 10, n_batches)[:,np.newaxis], (1, 9))
            drift_chan = drift_chan.astype('float32')
        plt.plot(drift_chan);
        plt.show()
    else:
        drift_chan = 10 * np.ones((n_batches, 9), 'float32')

    # upsample to get drift for all channels
    f = interp1d(np.linspace(0, 384, drift_chan.shape[1]+2)[1:-1], drift_chan, axis=-1, fill_value='extrapolate')
    drift_sim_ups = f(np.arange(0,384))
    drift_sim_ups = np.floor(drift_sim_ups * ups).astype(int)
    print(drift_sim_ups.min(), drift_sim_ups.max())
    print('created drift')

    # create spikes
    n_batches = int(np.ceil((st_sim.max() + n_twav) / batch_size))
    data = np.zeros((batch_size * n_batches, 385), 'float32')
    st_batch = np.floor(st_sim / batch_size).astype('int') # batch of each spike
    n_splits = 3
    tic=time.time()
    # loop over neurons
    print('creating data')
    for i in range(n_sim + n_noise):
        # waveform and channel for waveform
        wfi = wf_sim[i]
        ic0 = cb_sim[i] - nc//2
        iw0 = max(0, -ic0)
        ic1 = min(383, ic0+nc)
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
                for k in range(n_splits):
                    ki = np.arange(k, len(inds), n_splits)
                    # indices for spike waveform
                    stb = sti[inds[ki]] - int(twav_min)
                    stb = stb[:,np.newaxis].astype(int) + np.arange(0, n_twav, 1, int)
                    data[stb, ic0:ic1] += wfi[d][:, iw0:iw1]
                        
        if i%250==0:
            print(i, time.time()-tic)

    return data, st_sim, cl_sim, amp_sim, wf_sim, cb_sim, drift_sim_ups

def create_simulation(filename, st, cl, wfs, wfs_x, contaminations,
                      n_sim=100, n_noise=1000, n_batches=500,
                      batch_size=60000, tsig=50, tpad=100, n_chan_bin=385, drift=True,
                      drift_range=5, drift_seed=0, ups=10, whiten_mat = None,
                             ):
    """ simulate neuropixels 3B probe recording """

    # simulate spikes
    data, st_sim, cl_sim, amp_sim, wf_sim, cb_sim, drift_sim_ups = generate_spikes(st, cl, wfs, wfs_x, 
                                                                          contaminations, n_sim=n_sim, n_noise=n_noise, 
                                                                          n_batches=n_batches, tsig=tsig, 
                                                                          batch_size=batch_size, drift=drift,
                                                                          drift_range=drift_range, drift_seed=drift_seed,
                                                                           ups=ups)

    print(f'created dataset of size {data.shape}')
    
    # simulating background
    tweight = torch.arange(0, tpad, dtype=torch.float32)
    tweight_flip = tweight.flip(dims=(0,))
    tweight /= tweight + tweight_flip
    tweight = tweight.unsqueeze(1)
    tweight_flip = tweight.flip(dims=(0,))
    noise_init = generate_background(batch_size + tpad)
    noise_b = torch.zeros((batch_size, n_chan_bin))

    # writing binary
    with io.BinaryRWFile(filename, n_chan_bin, batch_size=batch_size, write=True) as bfile:
        for ibatch in np.arange(0, n_batches):
            noise_next = generate_background(batch_size + tpad)
            noise_pad = tweight_flip * noise_init[-tpad:] +  tweight * noise_next[:tpad]
            noise = torch.cat((noise_pad, noise_next[tpad : -tpad]), axis=0)
            noise_b[:, :384] = noise
            X = torch.from_numpy(data[ibatch * batch_size : min(n_batches * batch_size, (ibatch+1) * batch_size)]).float()
            
            X = (noise_b + X)

            if whiten_mat is not None:
                X[:,:384] = torch.linalg.solve(whiten_mat.cpu(), X[:,:384].T).T

            X = X.numpy()
            to_write = np.clip(20 * X, -2**15 + 1, 2**15 - 1).astype('int16')
            #to_write = np.clip(X, -2**15 + 1, 2**15 - 1).astype('int16')

            if ibatch%100==0:
                print(f'writing batch {ibatch} out of {n_batches}')
            bfile.write(to_write)
            noise_next = noise_init

    return st_sim, cl_sim, amp_sim, wf_sim, cb_sim, drift_sim_ups