import time
from pathlib import Path

import numpy as np
import torch

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
    """Get default settings dict for `run_kilosort`.

    Returns
    -------
    settings : dict
        A dictionary of parameters with the following keys:
        settings = {
            'NchanTOT'  : Total number of channels on probe.
            'fs'        : Sampling frequency of probe.
            'nt'        : Number of samples per waveform.
            'Th'        : TODO, some other threshold.
            'spkTh'     : Spike detection threshold for learned templates.
            'Th_detect' : Spike detection threshold for universal templates.
            'nwaves'    : Number of universal templates to use.
            'nskip'     : Batch striding for computing whitening matrix.
            'nt0min'    : Sample index for aligning waveforms, so that their
                          minimum value happens here. Default of 20 roughly
                          aligns to the sodium peak.
            'NT'        : Number of samples included in each batch of data.
            'nblocks'   : Number of non-overlapping blocks for drift correction.
            'binning_depth' : TODO, something to do with drift correction.
            'sig_interp': Sigma for interpolation (spatial standard deviation).
                          Indicates scale of waveform's smoothness for drift
                          correction, in units of microns.
            'n_chan_bin': Same as NchanTOT.
                          TODO: Why use both?
            'probe_name': Name of probe to load from Kilosort4's probe directory.
                          This will only be used if no `probe` kwarg is specified
                          for `run_kilosort`.
        }
    
    """
    settings = {}
    settings['NchanTOT']      = 385
    settings['fs']            = 30000
    settings['nt']            = 61
    settings['Th']            = 8
    settings['spkTh']         = 8
    settings['Th_detect']     = 9
    settings['nwaves']        = 6
    settings['nskip']         = 25
    settings['nt0min']        = int(20 * settings['nt']/61)
    settings['NT']            = 2 * settings['fs']
    settings['nblocks']       = 5
    settings['binning_depth'] = 5
    settings['sig_interp']    = 20
    settings['n_chan_bin']    = settings['NchanTOT']
    settings['probe_name']    = 'neuropixPhase3B1_kilosortChanMap.mat'
    return settings

def run_kilosort(settings=None, probe=None, probe_name=None, data_dir=None,
                 filename=None, data_dtype=None, results_dir=None, do_CAR=True,
                 device=torch.device('cuda'), progress_bar=None):

    if data_dtype is None:
        print("Interpreting binary file as default dtype='int16'. If data was "
              "saved in a different format, specify `data_dtype`.")

    if not do_CAR:
        print("Skipping common average reference.")

    tic0 = time.time()

    # Configure settings, ops, and file paths
    if settings is None: settings = default_settings()
    filename, data_dir, results_dir, probe = \
        set_files(settings, filename, probe, probe_name, data_dir, results_dir)
    ops = initialize_ops(settings, probe, data_dtype)

    # Set preprocessing and drift correction parameters
    ops = compute_preprocessing(ops, device, tic0=tic0)
    np.random.seed(1)
    torch.cuda.manual_seed_all(1)
    torch.random.manual_seed(1)
    ops, bfile = compute_drift_correction(ops, device, tic0=tic0,
                                          progress_bar=progress_bar)
    
    # Save intermediate `ops` for use by GUI plots
    save_ops(ops, results_dir)

    # Sort spikes and save results
    st, clu, tF, Wall = sort_spikes(ops, device, bfile, tic0=tic0,
                                    progress_bar=progress_bar)
    ops, similar_templates, is_ref, est_contam_rate = \
        save_sorting(ops, results_dir, st, clu, tF, Wall, tic0)

    return ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate


def set_files(settings, filename, probe, probe_name, data_dir, results_dir):
    """Parse file and directory information for data, probe, and results."""

    # Check for filename 
    filename = settings.get('filename', None) if filename is None else filename 

    # Use data_dir if filename not available
    if filename is None:
        data_dir = settings.get('data_dir', None) if data_dir is None else data_dir
        if data_dir is None:
            raise ValueError('no path to data provided, set "data_dir=" or "filename="')
        data_dir = Path(data_dir).resolve()
        if not data_dir.exists():
            raise FileExistsError(f"data_dir '{data_dir}' does not exist")

        # Find binary file in the folder
        filename  = io.find_binary(data_dir=data_dir)
        print(f"sorting {filename}")
    else:
        filename = Path(filename)
        if not filename.exists():
            raise FileExistsError(f"filename '{filename}' does not exist")
        data_dir = filename.parent
        
    # Convert paths to strings when saving to ops, otherwise ops can only
    # be loaded on the operating system that originally ran the code.
    settings['filename'] = filename
    settings['data_dir'] = data_dir

    results_dir = settings.get('results_dir', None) if results_dir is None else results_dir
    results_dir = Path(results_dir).resolve() if results_dir is not None else None
    
    # find probe configuration file and load
    if probe is None:
        if probe_name is not None:     probe_path = PROBE_DIR / probe_name
        elif 'probe_name' in settings: probe_path = PROBE_DIR / settings['probe_name']
        elif 'probe_path' in settings: probe_path = Path(settings['probe_path']).resolve()
        else: raise ValueError('no probe_name or probe_path provided, set probe_name=')
        if not probe_path.exists():
            raise FileExistsError(f"probe_path '{probe_path}' does not exist")
        
        probe  = io.load_probe(probe_path)
        print(f"using probe {probe_path.name}")

    return filename, data_dir, results_dir, probe


def initialize_ops(settings, probe, data_dtype) -> dict:
    """Package settings and probe information into a single `ops` dictionary."""

    # TODO: Clean this up during refactor. Lots of confusing duplication here.
    ops = settings  
    ops['settings'] = settings 
    ops['probe'] = probe
    ops['data_dtype'] = data_dtype
    ops['do_CAR'] = do_CAR
    ops['NTbuff'] = ops['NT'] + 2 * ops['nt']
    ops['Nchan'] = len(probe['chanMap'])
    ops['NchanTOT'] = ops['settings']['n_chan_bin']
    ops = {**ops, **probe}

    return ops

def get_run_parameters(ops) -> list:
    """Get `ops` dict values needed by `run_kilosort` subroutines."""

    parameters = [
        ops['settings']['n_chan_bin'],
        ops['settings']['fs'],
        ops['settings']['NT'],
        ops['settings']['nt'],
        ops['settings']['nt0min'],  # also called twav_min
        ops['probe']['chanMap'],
        ops['data_dtype'],
        ops['do_CAR'],
        ops['probe']['xc'],
        ops['probe']['yc']
    ]

    return parameters


def compute_preprocessing(ops, device, tic0=np.nan):
    """Compute preprocessing parameters and save them to `ops`.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    device : torch.device
        Indicates whether `pytorch` operations should be run on cpu or gpu.
    tic0 : float; default=np.nan
        Start time of `run_kilosort`.

    Returns
    -------
    ops : dict
    
    """

    tic = time.time()
    n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, xc, yc = \
        get_run_parameters(ops)
    nskip = ops['settings']['nskip']
    
    # Compute high pass filter
    hp_filter = preprocessing.get_highpass_filter(ops['settings']['fs'], device=device)
    # Compute whitening matrix
    bfile = io.BinaryFiltered(ops['filename'], n_chan_bin, fs, NT, nt, twav_min,
                              chan_map, hp_filter, device=device, do_CAR=do_CAR,
                              dtype=dtype)
    whiten_mat = preprocessing.get_whitening_matrix(bfile, xc, yc, nskip=nskip)

    bfile.close()

    # Save results
    ops['Nbatches'] = bfile.n_batches
    ops['preprocessing'] = {}
    ops['preprocessing']['whiten_mat'] = whiten_mat
    ops['preprocessing']['hp_filter'] = hp_filter
    ops['Wrot'] = whiten_mat
    ops['fwav'] = hp_filter

    print(f'Preprocessing filters computed in {time.time()-tic : .2f}s; ' +
            f'total {time.time()-tic0 : .2f}s')

    return ops


def compute_drift_correction(ops, device, tic0=np.nan, progress_bar=None):
    """Compute drift correction parameters and save them to `ops`.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    device : torch.device
        Indicates whether `pytorch` operations should be run on cpu or gpu.
    tic0 : float; default=np.nan.
        Start time of `run_kilosort`.
    progress_bar : TODO; optional.
        Informs `tqdm` package how to report progress, type unclear.

    Returns
    -------
    ops : dict
    bfile : kilosort.io.BinaryFiltered
        Wrapped file object for handling data.
    
    """
    tic = time.time()
    print('\ncomputing drift')

    n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, _, _ = \
        get_run_parameters(ops)
    hp_filter = ops['preprocessing']['hp_filter']
    whiten_mat = ops['preprocessing']['whiten_mat']

    bfile = io.BinaryFiltered(ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
                              hp_filter=hp_filter, whiten_mat=whiten_mat,
                              device=device, do_CAR=do_CAR, dtype=data_dtype)

    ops         = datashift.run(ops, bfile, device=device, progress_bar=progress_bar)
    bfile.close()
    print(f'drift computed in {time.time()-tic : .2f}s; ' + 
            f'total {time.time()-tic0 : .2f}s')
    
    # binary file with drift correction
    bfile = io.BinaryFiltered(ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
                              hp_filter=hp_filter, whiten_mat=whiten_mat,
                              dshift=ops['dshift'], do_CAR=do_CAR, dtype=dtype)

    return ops, bfile


def sort_spikes(ops, device, bfile, tic0=np.nan, progress_bar=None):
    """Run spike sorting algorithm and save intermediate results to `ops`.
    
    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    device : torch.device
        Indicates whether `pytorch` operations should be run on cpu or gpu.
    bfile : kilosort.io.BinaryFiltered
        Wrapped file object for handling data.
    tic0 : float; default=np.nan.
        Start time of `run_kilosort`.
    progress_bar : TODO; optional.
        Informs `tqdm` package how to report progress, type unclear.

    Returns
    -------
    st : np.ndarray
        1D vector of spike times for all clusters.
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st`.
    tF : np.ndarray
        TODO
    Wall : np.ndarray
        TODO

    """

    tic = time.time()
    print(f'\nExtracting spikes using built-in templates')
    st0, tF, ops  = spikedetect.run(ops, bfile, device=device, progress_bar=progress_bar)
    tF          = torch.from_numpy(tF)
    print(f'{len(st0)} spikes extracted in {time.time()-tic : .2f}s; ' + 
            f'total {time.time()-tic0 : .2f}s')

    tic = time.time()
    print('\nFirst clustering')
    clu, Wall   = clustering_qr.run(ops, st0, tF, mode = 'spikes', progress_bar=progress_bar)
    Wall3       = template_matching.postprocess_templates(Wall, ops, clu, st0, device=device)
    print(f'{clu.max()+1} clusters found, in {time.time()-tic : .2f}s; ' +
            f'total {time.time()-tic0 : .2f}s')
    
    tic = time.time()
    print('\nExtracting spikes using cluster waveforms')
    st, tF, tF2, ops = template_matching.extract(ops, bfile, Wall3, device=device, progress_bar=progress_bar)
    print(f'{len(st)} spikes extracted in {time.time()-tic : .2f}s; ' +
            f'total {time.time()-tic0 : .2f}s')

    tic = time.time()
    print('\nFinal clustering')
    clu, Wall   = clustering_qr.run(ops, st, tF,  mode = 'template', device=device, progress_bar=progress_bar)
    print(f'{clu.max()+1} clusters found, in {time.time()-tic : .2f}s; ' + 
            f'total {time.time()-tic0 : .2f}s')

    tic = time.time()
    print('\nMerging clusters')
    Wall, clu, is_ref = template_matching.merging_function(ops, Wall, clu, st[:,0])
    clu = clu.astype('int32')
    print(f'{clu.max()+1} units found, in {time.time()-tic : .2f}s; ' + 
            f'total {time.time()-tic0 : .2f}s')

    bfile.close()

    return st, clu, tF, Wall


def save_sorting(ops, results_dir, st, clu, tF, Wall, tic0=np.nan):  
    """Save sorting results, and format them for use with Phy

    Parameters
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    results_dir : pathlib.Path
        Directory where results should be saved.
    st : np.ndarray
        1D vector of spike times for all clusters.
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st`.
    tF : np.ndarray
        TODO
    Wall : np.ndarray
        TODO
    tic0 : float; default=np.nan.
        Start time of `run_kilosort`.

    Returns
    -------
    ops : dict
    similar_templates : np.ndarray
    is_ref : np.ndarray
    est_contam_rate : np.ndarray
    
    """

    print('\nSaving to phy and computing refractory periods')
    results_dir, similar_templates, is_ref, est_contam_rate = io.save_to_phy(
            st, clu, tF, Wall, ops['probe'], ops, results_dir=results_dir,
            data_dtype=ops['data_dtype']
            )
    print(f'{int(is_ref.sum())} units found with good refractory periods')
    
    ops['settings']['results_dir'] = str(results_dir)
    runtime = time.time()-tic0
    print(f'\nTotal runtime: {runtime:.2f}s = {int(runtime//3600):02d}:' +
          f'{int(runtime//60):02d}:{int(runtime%60)} h:m:s')
    ops['runtime'] = runtime 
    ops_arr = np.array(ops)
    
    # Convert paths to strings before saving, otherwise ops can only be loaded
    # on the system that originally ran the code (causes problems for tests).
    # TODO: why do these get saved twice?
    ops['filename'] = str(ops['filename'])
    ops['data_dir'] = str(ops['data_dir'])
    ops['settings']['filename'] = str(ops['settings']['filename'])
    ops['settings']['data_dir'] = str(ops['settings']['data_dir'])
    np.save(results_dir / 'ops.npy', ops_arr)

    return ops, similar_templates, is_ref, est_contam_rate


def save_ops(ops, results_dir):
    """Save intermediate `ops` dictionary to `results_dir/ops.npy`."""
    if results_dir is None:
        results_dir = ops['data_dir'].joinpath('kilosort4')
    results_dir.mkdir(exist_ok=True)
    np.save(results_dir / 'ops.npy', np.array(ops))
