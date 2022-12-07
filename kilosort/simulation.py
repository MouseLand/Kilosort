import numpy as np 
import os, time 
import matplotlib.pyplot as plt 
import torch 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from kilosort import io

def generate_background(NT, whiten_mat, fs=30000, device=torch.device('cuda')):
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
                    batch_size=60000, twav_min=50, drift=True):
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
    amp_sim = np.random.exponential(scale=1., size=(n_sim,)) * 12. + 12.
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
            fr = len(spi) / n_time_sim
            #if fr > 1:
            #    spi = spi[np.random.rand(len(spi)) < frac_rand]

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
    frs *= 10
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
        #if fr > 1:
        #    spi = spi[np.random.rand(len(spi)) < frac_rand]

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
    
    # normalize and amplitude multiply waveforms
    #wf_sim /= np.abs(wf_sim).max()
    #wf_sim *= int(5000 / amp_sim.max())
    wf_sim *= amp_sim[:,np.newaxis,np.newaxis,np.newaxis]
    #wf_sim = wf_sim.astype('int16')
    #print(wf_sim.max())
    
    np.random.seed(0)
    # overall drift
    if drift:
        drift_sim = np.random.randn(n_batches)
        drift_sim = gaussian_filter1d(drift_sim, tsig)
        drift_sim -= drift_sim.min()
        drift_sim /= drift_sim.max()
        drift_sim *= 20
        
        # drift across probe (9 positions)
        drift_chan = np.random.randn(n_batches, 9)
        drift_chan = gaussian_filter1d(drift_chan, tsig, axis=0)
        drift_chan -= drift_chan.min(axis=0) 
        drift_chan /= drift_chan.max(axis=0)
        drift_chan *= 20
        drift_chan = gaussian_filter1d(drift_chan*4, 2, axis=-1)
        drift_chan += drift_sim[:,np.newaxis]
        drift_chan -= drift_chan.min() 
        drift_chan /= drift_chan.max()
        drift_chan *= 20

        plt.plot(drift_chan);
        plt.show()
    else:
        drift_chan = 10 * np.ones((n_batches, 9), 'float32')

    # upsample to get drift for all channels
    f = interp1d(np.linspace(0, 384, drift_chan.shape[1]+2)[1:-1], drift_chan, axis=-1, fill_value='extrapolate')
    drift_sim_ups = f(np.arange(0,384))
    drift_sim_ups = np.floor(drift_sim_ups).astype(int)
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
        #print(cb_sim[i] - nc//2, ic0, ic1, iw0, iw1)
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

def create_simulation(filename, whiten_mat_dat, st, cl, wfs, wfs_x, contaminations,
                      n_sim=100, n_noise=1000, n_batches=500,
                      batch_size=60000, tsig=50, tpad=100, n_chan_bin=385, drift=True
                             ):
    """ simulate neuropixels 3B probe recording """

    # simulate spikes
    data, st_sim, cl_sim, amp_sim, wf_sim, cb_sim, drift_sim_ups = generate_spikes(st, cl, wfs, wfs_x, 
                                                                          contaminations, n_sim=n_sim, n_noise=n_noise, 
                                                                          n_batches=n_batches, tsig=tsig, 
                                                                          batch_size=batch_size, drift=drift)

    # simulating background
    tweight = torch.arange(0, tpad, dtype=torch.float32)
    tweight_flip = tweight.flip(dims=(0,))
    tweight /= tweight + tweight_flip
    tweight = tweight.unsqueeze(1)
    tweight_flip = tweight.flip(dims=(0,))
    noise_init = generate_background(batch_size + tpad, whiten_mat_dat)
    #nfactor = (data[::1000,::5].std() * 2.5)
    #noise_init *= nfactor
    noise_b = torch.zeros((batch_size, n_chan_bin))

    # writing binary
    with io.BinaryRWFile(filename, n_chan_bin, batch_size=batch_size, write=True) as bfile:
        for ibatch in np.arange(0, n_batches):
            noise_next = generate_background(batch_size + tpad, whiten_mat_dat)
            noise_pad = tweight_flip * noise_init[-tpad:] +  tweight * noise_next[:tpad]
            noise = torch.cat((noise_pad, noise_next[tpad : -tpad]), axis=0)
            noise_b[:, :384] = noise
            X = torch.from_numpy(data[ibatch * batch_size : min(n_batches * batch_size, (ibatch+1) * batch_size)]).float()
            #X[:,:384] = torch.linalg.solve(whiten_mat_dat.cpu(), X[:,:384].T).T
            X = (noise_b + X).numpy()
            #X = X.numpy()
            to_write = np.clip(200 * X, -2**15 + 1, 2**15 - 1).astype('int16')
            if ibatch%100==0:
                print(ibatch, to_write.shape)
            #print(to_write.max(), to_write.min())
            bfile.write(to_write)
            noise_next = noise_init

    return st_sim, cl_sim, amp_sim, wf_sim, cb_sim, drift_sim_ups