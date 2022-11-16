import os
from glob import glob
import numpy as np
import torch
from kilosort import preprocessing, datashift, template_matching, clustering_qr, CCG, clustering_qr, io
#from kilosort import default_settings


def run_kilosort(settings=None, probe=None, data_folder=None, device=torch.device('cuda')):
    if settings is None:
        settings = default_settings()
        if data_folder is None:
            raise ValueError('no path to data provided, set "data_folder="')
        settings['data_folder'] = data_folder

    # find probe configuration file and load
    if probe is None: 
        probe  = io.load_probe(settings['probe_path'])

    ops = {}
    ops['settings'] = settings 
    ops['probe'] = probe
    #{'settings': settings,
          # 'probe': probe}

    # find binary file in the folder
    filename  = io.find_binary(data_folder=settings['data_folder'])

    print(f"sorting {filename}")
    ops['filename'] = filename 
    import time
    tic = time.time()
    
    nt = ops['settings']['nt']
    NT = ops['settings']['NT']
    fs = ops['settings']['fs']
    NchanTOT = ops['settings']['n_chan_bin']
    twav_min = ops['settings']['nt0min']
    n_wavpc = ops['settings']['nwaves']
    
    channel_map = ops['probe']['chanMap']
    x_chan = ops['probe']['xc']
    y_chan = ops['probe']['yc']
    
    # compute high pass filter
    hp_filter = preprocessing.get_highpass_filter(ops['settings']['fs'], device=device)
    # compute whitening matrix
    with io.BinaryFiltered(filename, NchanTOT, fs, NT, nt, twav_min, channel_map,
                            hp_filter, None, None, device=device) as bfile:
        whiten_mat = preprocessing.get_whitening_matrix(bfile, x_chan, y_chan, 
                                                        nskip = ops['settings']['nskip'])
    print(time.time()-tic)

    #print(whiten_mat.dtype)
    
    binning_depth = ops['settings']['binning_depth']
    spike_threshold = ops['settings']['spkTh']
    init_threshold = ops['settings']['Th']
    nblocks = ops['settings']['nblocks']
    sig_interp = ops['settings']['sig_interp']

    # compute z-drift
    with io.BinaryFiltered(filename, NchanTOT, fs, NT, nt, twav_min, channel_map,
                            hp_filter, whiten_mat, None, device=device) as bfile:
        ops  = datashift.run(bfile, x_chan, y_chan, init_threshold, n_wavpc, binning_depth,
                             spike_threshold, nblocks, sig_interp, ops)
    print(time.time()-tic)

    # find spikes
    with io.BinaryFiltered(filename, NchanTOT, fs, NT, nt, twav_min, channel_map,
                            hp_filter, whiten_mat, ops['dshift'], device=device) as bfile:
        ops['U']         = template_matching.run(bfile, ops)
        st, tF, tF2, ops = template_matching.extract(bfile, ops, ops['U'])

    # cluster spikes
    iclust, Wall  = clustering_qr.run(ops, st, tF)

    # quality control
    is_refractory = CCG.refract(iclust, st[:,0])

    return ops, st, tF, iclust, Wall, is_refractory