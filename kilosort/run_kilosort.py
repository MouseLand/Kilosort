import os
from glob import glob
import numpy as np
import torch
from kilosort import (
    preprocessing,
    datashift,
    template_matching,
    clustering_qr,
    CCG,
    clustering_qr,
    io,
    spikedetect,
)
#from kilosort import default_settings


def run_kilosort(settings=None, probe=None, data_folder=None, device=torch.device('cuda')):
    if settings is None:
        settings = {}
        #settings = default_settings()
        if data_folder is None:
            raise ValueError('no path to data provided, set "data_folder="')
        settings['data_folder'] = data_folder

    if not os.path.exists(settings['data_folder']):
        return FileExistsError(f"data_folder '{settings['data_folder']}' does not exist")

    # find probe configuration file and load
    if probe is None: 
        probe  = io.load_probe(settings['probe_path'])

    ops = {}
    ops = settings  
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

    ops['NTbuff'] = ops['NT'] + 2 * ops['nt']

    channel_map = probe['chanMap']
    x_chan = probe['xc']
    y_chan = probe['yc']
    ops['Nchan'] = len(probe['chanMap'])
    ops['NchanTOT'] = NchanTOT
    ops = {**ops, **probe}

    # compute high pass filter
    hp_filter = preprocessing.get_highpass_filter(ops['settings']['fs'], device=device)
    # compute whitening matrix
    with io.BinaryFiltered(filename, NchanTOT, fs, NT, nt, twav_min, channel_map,
                            hp_filter, None, None, device=device) as bfile:
        whiten_mat = preprocessing.get_whitening_matrix(bfile, x_chan, y_chan, 
                                                        nskip = ops['settings']['nskip'])
        ops['Nbatches'] = bfile.n_batches

    print(time.time()-tic)

    #print(whiten_mat.dtype)
    
    np.random.seed(1)
    torch.cuda.manual_seed_all(1)
    torch.random.manual_seed(1)

    ops          = preprocessing.run(ops, device=device)
    #ops['Nbatches'] = 200

    ops         = datashift.run(ops, device=device)

    st0, tF, ops  = spikedetect.run(ops, dshift=ops['dshift'], device=device)

    tF          = torch.from_numpy(tF)

    clu, Wall   = clustering_qr.run(ops, st0, tF, mode = 'spikes') 

    Wall3       = template_matching.postprocess_templates(Wall, ops, clu, st0, device=device)

    st, tF, tF2, ops = template_matching.extract(ops, Wall3, device=device)

    clu, Wall   = clustering_qr.run(ops, st, tF,  mode = 'template', device=device)

    #is_ref = CCG.refract(clu, st[:,0])

    Wall, clu, is_ref = template_matching.merging_function(ops, Wall, clu, st[:,0])

    is_ref, est_contam_rate = CCG.refract(clu, st/ops['fs'])

    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])
    mu = (Wall**2).sum((1,2), keepdims=True)**.5
    Wnorm = Wall / (1e-6 + mu)
    UtU = torch.einsum('ilk, jlm -> ijkm',  Wnorm, Wnorm)
    similar_templates = torch.einsum('ijkm, kml -> ijl', UtU, WtW)



    return ops, st, tF, clu, Wall, is_ref, est_contam_rate