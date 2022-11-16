import os
from glob import glob
import numpy as np
from scipy.io import loadmat
import torch
from kilosort import preprocessing, datashift, template_matching, clustering_qr, CCG, clustering_qr, io
#from kilosort import default_settings

def find_file(data_folder, n_chan_tot=385, batch_size=60000):
    """ find binary file in data_folder"""
    filename  = glob(os.path.join(data_folder, '*.bin'))[0]
    return filename

def load_probe(probe_path):
    """add configuration options from the matlab probe files
    """
    dconfig     = loadmat(probe_path)
    
    connected = dconfig['connected'].astype('bool')

    probe = {}
    probe['Nchan']     = connected.sum()

    probe['chanMap']   = dconfig['chanMap'][connected].astype('int16').flatten() - 1
    probe['xc']        = dconfig['xcoords'][connected].astype('float32')
    probe['yc']        = dconfig['ycoords'][connected].astype('float32')

    return probe

def run_kilosort(settings=None, data_folder=None, device=torch.device('cuda')):
    if settings is None:
        settings = default_settings()
        if data_folder is None:
            raise ValueError('no path to data provided, set "data_folder="')
        settings['data_folder'] = data_folder

    # find probe configuration file and load 
    probe  = load_probe(settings['probe_path'])

    ops = {}
    ops['settings'] = settings 
    ops['probe'] = probe
    #{'settings': settings,
          # 'probe': probe}

    # find binary file in the folder
    filename  = find_file(data_folder=settings['data_folder'],
                                     n_chan_tot=settings['NchanTOT'],#['n_chan_tot'],
                                     batch_size=settings['NT'])#['batch_size'])
    print(f"sorting {filename}")
    ops['filename'] = filename 
    import time
    tic = time.time()
    
    #ops  = preprocessing.run(ops)
    hp_filter = preprocessing.get_highpass_filter(ops['settings']['fs'], device=device)
    # compute whitening matrix
    nt = ops['settings']['nt']
    NT = ops['settings']['NT']
    fs = ops['settings']['fs']
    NchanTOT = ops['settings']['NchanTOT']
    twav_min = ops['settings']['nt0min']
    n_wavpc = ops['settings']['nwaves']
    
    channel_map = ops['probe']['chanMap']
    x_chan = ops['probe']['xc']
    y_chan = ops['probe']['yc']
    with io.BinaryFiltered(filename, NchanTOT, fs, NT, nt, twav_min, channel_map,
                            hp_filter, None, None) as bfile:
        whiten_mat = preprocessing.get_whitening_matrix(bfile, x_chan, y_chan, 
                                                        nskip = ops['settings']['nskip'])
    print(time.time()-tic)
    
    binning_depth = ops['settings']['binning_depth']
    spike_threshold = ops['settings']['spkTh']
    init_threshold = ops['settings']['Th']
    nblocks = ops['settings']['nblocks']
    sig_interp = ops['settings']['sig_interp']
    with io.BinaryFiltered(filename, NchanTOT, fs, NT, nt, twav_min, channel_map,
                            hp_filter, whiten_mat, None) as bfile:
        ops  = datashift.run(bfile, x_chan, y_chan, init_threshold, n_wavpc, binning_depth,
                             spike_threshold, nblocks, sig_interp, ops)
    print(time.time()-tic)

    with io.BinaryFiltered(filename, NchanTOT, fs, NT, nt, twav_min, channel_map,
                            hp_filter, whiten_mat, ops['dshift']) as bfile:
        ops['U']         = template_matching.run(bfile, ops)
        st, tF, tF2, ops = template_matching.extract(bfile, ops, ops['U'])

    iclust, Wall  = clustering_qr.run(ops, st, tF)
    is_refractory = CCG.refract(iclust, st[:,0])

    return ops