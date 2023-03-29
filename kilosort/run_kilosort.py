import time
import numpy as np
import torch
import tqdm
from pathlib import Path
from kilosort import (
    preprocessing,
    datashift,
    template_matching,
    clustering_qr,
    clustering_qr,
    io,
    spikedetect,
    PROBE_DIR
)

def default_settings():
    settings = {}
    settings['NchanTOT'] = 385
    settings['fs']       = 30000
    settings['nt']     =  61
    settings['Th']       = 8
    settings['spkTh']    = 8
    settings['Th_detect']    = 9
    settings['nwaves']   = 6
    settings['nskip']    = 25
    settings['nt0min']   = int(20 * settings['nt']/61)
    settings['NT']       = 2 * settings['fs']
    settings['nblocks']  = 5
    settings['binning_depth'] = 5
    settings['sig_interp'] = 20
    settings['n_chan_bin'] = settings['NchanTOT']
    settings['probe_name'] = 'neuropixPhase3B1_kilosortChanMap.mat'
    return settings

def run_kilosort(settings=None, probe=None, probe_name=None, data_dir=None, filename=None,
                 results_dir=None, device=torch.device('cuda'), progress_bar=None,
                 save=True):

    tic0 = time.time()

    if settings is None:
        settings = default_settings()

    # check for filename 
    filename = settings.get('filename', None) if filename is None else filename 

    # use data_dir if filename not available
    if filename is None:
        data_dir = settings.get('data_dir', None) if data_dir is None else data_dir
        if data_dir is None:
            raise ValueError('no path to data provided, set "data_dir=" or "filename="')
        data_dir = Path(data_dir).resolve()
        if not data_dir.exists():
            raise FileExistsError(f"data_dir '{data_dir}' does not exist")
        # find binary file in the folder
        filename  = io.find_binary(data_dir=data_dir) if 'filename' not in settings else settings['filename']
        print(f"sorting {filename}")
    else:
        filename = Path(filename)
        if not filename.exists():
            raise FileExistsError(f"filename '{filename}' does not exist")
        data_dir = filename.parent
        
    settings['filename'] = filename 
    settings['data_dir'] = data_dir

    results_dir = settings.get('results_dir', None) if results_dir is None else results_dir
    results_dir = Path(results_dir).resolve() if results_dir is not None else None
    
    # find probe configuration file and load
    if probe is None: 
        if probe_name is not None or 'probe_name' in settings:
            probe_path = PROBE_DIR / probe_name if probe_name is not None else PROBE_DIR / settings['probe_name']
        elif 'probe_path' in settings:
            probe_path = Path(settings['probe_path']).resolve()
        else:
            raise ValueError('no probe_name or probe_path provided, set probe_name=')
        if not probe_path.exists():
            raise FileExistsError(f"probe_path '{probe_path}' does not exist")
        probe  = io.load_probe(probe_path)
        print(f"using probe {probe_path.name}")
    
    ops = {}
    ops = settings  
    ops['settings'] = settings 
    ops['probe'] = probe
    #{'settings': settings,
            # 'probe': probe}

    
    nt = ops['settings']['nt']
    NT = ops['settings']['NT']
    fs = ops['settings']['fs']
    n_chan_bin = ops['settings']['n_chan_bin']
    twav_min = ops['settings']['nt0min']
    n_wavpc = ops['settings']['nwaves']

    ops['NTbuff'] = ops['NT'] + 2 * ops['nt']

    chan_map = probe['chanMap']
    xc = probe['xc']
    yc = probe['yc']
    ops['Nchan'] = len(probe['chanMap'])
    ops['NchanTOT'] = n_chan_bin
    ops = {**ops, **probe}

    ### preprocessing

    tic = time.time()
    # compute high pass filter
    hp_filter = preprocessing.get_highpass_filter(ops['settings']['fs'], device=device)
    # compute whitening matrix
    bfile = io.BinaryFiltered(filename, n_chan_bin, fs, NT, nt, twav_min, chan_map, hp_filter, device=device)
    whiten_mat = preprocessing.get_whitening_matrix(bfile, xc, yc, 
                                                    nskip = ops['settings']['nskip'])
    bfile.close()
    ops['Nbatches'] = bfile.n_batches
    ops['preprocessing'] = {}
    ops['preprocessing']['whiten_mat'] = whiten_mat
    ops['preprocessing']['hp_filter'] = hp_filter

    ops['Wrot'] = whiten_mat
    ops['fwav'] = hp_filter

    print(f'preprocessing filters computed in {time.time()-tic : .2f}s; total {time.time()-tic0 : .2f}s')

    np.random.seed(1)
    torch.cuda.manual_seed_all(1)
    torch.random.manual_seed(1)

    ### drift computation
    print('\ncomputing drift')
    tic = time.time()
    bfile = io.BinaryFiltered(filename, n_chan_bin, fs, NT, nt, twav_min, chan_map, 
                              hp_filter=hp_filter, whiten_mat=whiten_mat, device=device)
    ops         = datashift.run(ops, bfile, device=device, progress_bar=progress_bar)
    bfile.close()
    print(f'drift computed in {time.time()-tic : .2f}s; total {time.time()-tic0 : .2f}s')
    
    # binary file with drift correction
    bfile = io.BinaryFiltered(filename, n_chan_bin, fs, NT, nt, twav_min, chan_map, 
                              hp_filter=hp_filter, whiten_mat=whiten_mat, dshift=ops['dshift'])

    ### spike sorting

    tic = time.time()
    print(f'\nextracting spikes using built-in templates')
    st0, tF, ops  = spikedetect.run(ops, bfile, device=device, progress_bar=progress_bar)
    tF          = torch.from_numpy(tF)
    print(f'{len(st0)} spikes extracted in {time.time()-tic : .2f}s; total {time.time()-tic0 : .2f}s')

    tic = time.time()
    print('\nfirst clustering')
    clu, Wall   = clustering_qr.run(ops, st0, tF, mode = 'spikes', progress_bar=progress_bar)
    Wall3       = template_matching.postprocess_templates(Wall, ops, clu, st0, device=device)
    print(f'{clu.max()+1} clusters found, in {time.time()-tic : .2f}s; total {time.time()-tic0 : .2f}s')
    
    tic = time.time()
    print('\nextracting spikes using cluster waveforms')
    st, tF, tF2, ops = template_matching.extract(ops, bfile, Wall3, device=device, progress_bar=progress_bar)
    print(f'{len(st)} spikes extracted in {time.time()-tic : .2f}s; total {time.time()-tic0 : .2f}s')

    tic = time.time()
    print('\nfinal clustering')
    clu, Wall   = clustering_qr.run(ops, st, tF,  mode = 'template', device=device, progress_bar=progress_bar)
    print(f'{clu.max()+1} clusters found, in {time.time()-tic : .2f}s; total {time.time()-tic0 : .2f}s')

    tic = time.time()
    print('\nmerging clusters')
    Wall, clu, is_ref = template_matching.merging_function(ops, Wall, clu, st[:,0])
    clu = clu.astype('int32')
    print(f'{clu.max()+1} units found, in {time.time()-tic : .2f}s; total {time.time()-tic0 : .2f}s')

    bfile.close()

    # NOTE: Generally speaking, `save=False` should only be used for testing.
    if save:
        print('\nsaving to phy and computing refractory periods')
        # save to phy and compute more properties of units
        results_dir, similar_templates, is_ref, est_contam_rate = io.save_to_phy(st, clu, tF, Wall, probe, ops, 
                                                                    results_dir=results_dir )
        print(f'{int(is_ref.sum())} units found with good refractory periods')
        
        ops['settings']['results_dir'] = results_dir
        runtime = time.time()-tic0 
        print(f'\ntotal runtime: {runtime:.2f}s = {int(runtime//3600):02d}:{int(runtime//60):02d}:{int(runtime%60)} h:m:s')
        ops['runtime'] = runtime 
        ops_arr = np.array(ops)
        
        np.save(results_dir / 'ops.npy', ops_arr)

    return ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate