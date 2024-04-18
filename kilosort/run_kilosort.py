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
    CCG,
    PROBE_DIR
)
from kilosort.parameters import DEFAULT_SETTINGS


def run_kilosort(settings, probe=None, probe_name=None, filename=None,
                 data_dir=None, file_object=None, results_dir=None,
                 data_dtype=None, do_CAR=True, invert_sign=False, device=None,
                 progress_bar=None, save_extra_vars=False):
    """Run full spike sorting pipeline on specified data.
    
    Parameters
    ----------
    settings : dict
        Specifies a number of configurable parameters used throughout the
        spike sorting pipeline. See `kilosort/parameters.py` for a full list of
        available parameters.
        NOTE: `n_chan_bin` must be specified here, but all other settings are
              optional.
    probe : dict; optional.
        A Kilosort4 probe dictionary, as returned by `kilosort.io.load_probe`.
    probe_name : str; optional.
        Filename of probe to use, within the default `PROBE_DIR`. Only include
        the filename without any preceeding directories. Will ony be used if
        `probe is None`. Alternatively, the full filepath to a probe stored in
        any directory can be specified with `settings = {'probe_path': ...}`.
        See `kilosort.utils` for default `PROBE_DIR` definition.
    filename: str or Path; optional.
        Full path to binary data file. If specified, will also set
        `data_dir = filename.parent`.
    data_dir : str or Path; optional.
        Specifies directory where binary data file is stored. Kilosort will
        attempt to find the binary file. This works best if there is exactly one
        file in the directory with a .bin, .bat, .dat, or .raw extension.
        Only used if `filename is None`.
        Also see `kilosort.io.find_binary`.
    file_object : array-like file object; optional.
        Must have 'shape' and 'dtype' attributes and support array-like
        indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
        array or memmap. Must specify a valid `filename` as well, even though
        data will not be directly loaded from that file.
    results_dir : str or Path; optional.
        Directory where results will be stored. By default, will be set to
        `data_dir / 'kilosort4'`.
    data_dtype : str or type; optional.
        dtype of data in binary file, like `'int32'` or `np.uint16`. By default,
        dtype is assumed to be `'int16'`.
    do_CAR : bool; default=True.
        If True, apply common average reference during preprocessing
        (recommended).
    invert_sign : bool; default=False.
        If True, flip positive/negative values in data to conform to standard
        expected by Kilosort4.
    device : torch.device; optional.
        CPU or GPU device to use for PyTorch calculations. By default, PyTorch
        will use the first detected GPU. If no GPUs are detected, CPU will be
        used. To set this manually, specify `device = torch.device(<device_name>)`.
        See PyTorch documentation for full description.
    progress_bar : tqdm.std.tqdm or QtWidgets.QProgressBar; optional.
        Used by sorting steps and GUI to track sorting progress. Users should
        not need to specify this.
    save_extra_vars : bool; default=False.
        If True, save tF and Wall to disk after sorting.
    
    Raises
    ------
    ValueError
        If settings[`n_chan_bin`] is None (default). User must specify, for
        example:  `run_kilosort(settings={'n_chan_bin': 385})`.

    Returns
    -------
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate
        Description TODO

    """

    if not do_CAR:
        print("Skipping common average reference.")

    tic0 = time.time()

    # Configure settings, ops, and file paths
    if settings is None or settings.get('n_chan_bin', None) is None:
        raise ValueError(
            '`n_chan_bin` is a required setting. This is the total number of '
            'channels in the binary file, which may or may not be equal to the '
            'number of channels specified by the probe.'
            )
    settings = {**DEFAULT_SETTINGS, **settings}

    if data_dtype is None:
        print("Interpreting binary file as default dtype='int16'. If data was "
                "saved in a different format, specify `data_dtype`.")

    if device is None:
        if torch.cuda.is_available():
            print('Using GPU for PyTorch computations. '
                  'Specify `device` to change this.')
            device = torch.device('cuda')
        else:
            print('Using CPU for PyTorch computations. '
                  'Specify `device` to change this.')
            device = torch.device('cpu')

    # NOTE: Also modifies settings in-place
    filename, data_dir, results_dir, probe = \
        set_files(settings, filename, probe, probe_name, data_dir, results_dir)
    ops = initialize_ops(settings, probe, data_dtype, do_CAR, invert_sign, device)

    if probe['chanMap'].max() >= settings['n_chan_bin']:
        raise ValueError(
            f'Largest value of chanMap exceeds channel count of data, '
             'make sure chanMap is 0-indexed.'
        )

    # Set preprocessing and drift correction parameters
    ops = compute_preprocessing(ops, device, tic0=tic0, file_object=file_object)
    np.random.seed(1)
    torch.cuda.manual_seed_all(1)
    torch.random.manual_seed(1)
    ops, bfile, st0 = compute_drift_correction(
        ops, device, tic0=tic0, progress_bar=progress_bar,
        file_object=file_object
        )
    
    # TODO: don't think we need to do this actually
    # Save intermediate `ops` for use by GUI plots
    io.save_ops(ops, results_dir)

    # Sort spikes and save results
    st, tF, _, _ = detect_spikes(ops, device, bfile, tic0=tic0,
                                 progress_bar=progress_bar)
    clu, Wall = cluster_spikes(st, tF, ops, device, bfile, tic0=tic0,
                               progress_bar=progress_bar)
    ops, similar_templates, is_ref, est_contam_rate, kept_spikes = \
        save_sorting(ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0,
                     save_extra_vars=save_extra_vars)

    return ops, st, clu, tF, Wall, similar_templates, \
           is_ref, est_contam_rate, kept_spikes


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


def initialize_ops(settings, probe, data_dtype, do_CAR, invert_sign, device) -> dict:
    """Package settings and probe information into a single `ops` dictionary."""

    if settings['nt0min'] is None:
        settings['nt0min'] = int(20 * settings['nt']/61)
    # TODO: Clean this up during refactor. Lots of confusing duplication here.
    ops = settings  
    ops['settings'] = settings 
    ops['probe'] = probe
    ops['data_dtype'] = data_dtype
    ops['do_CAR'] = do_CAR
    ops['invert_sign'] = invert_sign
    ops['NTbuff'] = ops['batch_size'] + 2 * ops['nt']
    ops['Nchan'] = len(probe['chanMap'])
    ops['n_chan_bin'] = settings['n_chan_bin']
    ops['torch_device'] = str(device)

    if not settings['templates_from_data'] and settings['nt'] != 61:
        raise ValueError('If using pre-computed universal templates '
                         '(templates_from_data=False), nt must be 61')

    ops = {**ops, **probe}

    return ops

def get_run_parameters(ops) -> list:
    """Get `ops` dict values needed by `run_kilosort` subroutines."""

    parameters = [
        ops['settings']['n_chan_bin'],
        ops['settings']['fs'],
        ops['settings']['batch_size'],
        ops['settings']['nt'],
        ops['settings']['nt0min'],  # also called twav_min
        ops['probe']['chanMap'],
        ops['data_dtype'],
        ops['do_CAR'],
        ops['invert_sign'],
        ops['probe']['xc'],
        ops['probe']['yc'],
        ops['settings']['tmin'],
        ops['settings']['tmax'],
        ops['settings']['artifact_threshold']
    ]

    return parameters


def compute_preprocessing(ops, device, tic0=np.nan, file_object=None):
    """Compute preprocessing parameters and save them to `ops`.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    device : torch.device
        Indicates whether `pytorch` operations should be run on cpu or gpu.
    tic0 : float; default=np.nan
        Start time of `run_kilosort`.
    file_object : array-like file object; optional.
        Must have 'shape' and 'dtype' attributes and support array-like
        indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
        array or memmap.

    Returns
    -------
    ops : dict
    
    """

    tic = time.time()
    n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
        xc, yc, tmin, tmax, artifact = get_run_parameters(ops)
    nskip = ops['settings']['nskip']
    whitening_range = ops['settings']['whitening_range']
    
    # Compute high pass filter
    hp_filter = preprocessing.get_highpass_filter(ops['settings']['fs'], device=device)
    # Compute whitening matrix
    bfile = io.BinaryFiltered(ops['filename'], n_chan_bin, fs, NT, nt, twav_min,
                              chan_map, hp_filter, device=device, do_CAR=do_CAR,
                              invert_sign=invert, dtype=dtype, tmin=tmin,
                              tmax=tmax, artifact_threshold=artifact,
                              file_object=file_object)
    whiten_mat = preprocessing.get_whitening_matrix(bfile, xc, yc, nskip=nskip,
                                                    nrange=whitening_range)

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


def compute_drift_correction(ops, device, tic0=np.nan, progress_bar=None,
                             file_object=None):
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
    file_object : array-like file object; optional.
        Must have 'shape' and 'dtype' attributes and support array-like
        indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
        array or memmap.

    Returns
    -------
    ops : dict
    bfile : kilosort.io.BinaryFiltered
        Wrapped file object for handling data.
    
    """
    tic = time.time()
    print('\ncomputing drift')

    n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
        _, _, tmin, tmax, artifact = get_run_parameters(ops)
    hp_filter = ops['preprocessing']['hp_filter']
    whiten_mat = ops['preprocessing']['whiten_mat']

    bfile = io.BinaryFiltered(
        ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
        hp_filter=hp_filter, whiten_mat=whiten_mat, device=device, do_CAR=do_CAR,
        invert_sign=invert, dtype=dtype, tmin=tmin, tmax=tmax,
        artifact_threshold=artifact, file_object=file_object
        )

    ops, st = datashift.run(ops, bfile, device=device, progress_bar=progress_bar)
    bfile.close()
    print(f'drift computed in {time.time()-tic : .2f}s; ' + 
            f'total {time.time()-tic0 : .2f}s')
    
    # binary file with drift correction
    bfile = io.BinaryFiltered(
        ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
        hp_filter=hp_filter, whiten_mat=whiten_mat, device=device,
        dshift=ops['dshift'], do_CAR=do_CAR, dtype=dtype, tmin=tmin, tmax=tmax,
        artifact_threshold=artifact, file_object=file_object
        )

    return ops, bfile, st


def detect_spikes(ops, device, bfile, tic0=np.nan, progress_bar=None):
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
    print(f'\nExtracting spikes using templates')
    st0, tF, ops = spikedetect.run(ops, bfile, device=device, progress_bar=progress_bar)
    tF = torch.from_numpy(tF)
    print(f'{len(st0)} spikes extracted in {time.time()-tic : .2f}s; ' + 
            f'total {time.time()-tic0 : .2f}s')
    if len(st0) == 0:
        raise ValueError('No spikes detected, cannot continue sorting.')

    tic = time.time()
    print('\nFirst clustering')
    clu, Wall = clustering_qr.run(ops, st0, tF, mode='spikes', device=device,
                                  progress_bar=progress_bar)
    Wall3 = template_matching.postprocess_templates(Wall, ops, clu, st0, device=device)
    print(f'{clu.max()+1} clusters found, in {time.time()-tic : .2f}s; ' +
            f'total {time.time()-tic0 : .2f}s')
    
    tic = time.time()
    print('\nExtracting spikes using cluster waveforms')
    st, tF, ops = template_matching.extract(ops, bfile, Wall3, device=device,
                                                 progress_bar=progress_bar)
    print(f'{len(st)} spikes extracted in {time.time()-tic : .2f}s; ' +
            f'total {time.time()-tic0 : .2f}s')

    return st, tF, Wall, clu


def cluster_spikes(st, tF, ops, device, bfile, tic0=np.nan, progress_bar=None):
    tic = time.time()
    print('\nFinal clustering')
    clu, Wall = clustering_qr.run(ops, st, tF,  mode = 'template', device=device,
                                  progress_bar=progress_bar)
    print(f'{clu.max()+1} clusters found, in {time.time()-tic : .2f}s; ' + 
            f'total {time.time()-tic0 : .2f}s')

    tic = time.time()
    print('\nMerging clusters')
    Wall, clu, is_ref = template_matching.merging_function(ops, Wall, clu, st[:,0],
                                                           device=device)
    clu = clu.astype('int32')
    print(f'{clu.max()+1} units found, in {time.time()-tic : .2f}s; ' + 
            f'total {time.time()-tic0 : .2f}s')

    bfile.close()

    return clu, Wall


def save_sorting(ops, results_dir, st, clu, tF, Wall, imin, tic0=np.nan,
                 save_extra_vars=False):  
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
    imin : int
        Minimum sample index used by BinaryRWFile, exported spike times will
        be shifted forward by this number.
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
    results_dir, similar_templates, is_ref, est_contam_rate, kept_spikes = \
        io.save_to_phy(
            st, clu, tF, Wall, ops['probe'], ops, imin, results_dir=results_dir,
            data_dtype=ops['data_dtype'], save_extra_vars=save_extra_vars
            )
    print(f'{int(is_ref.sum())} units found with good refractory periods')
    
    runtime = time.time()-tic0
    seconds = runtime % 60
    mins = runtime // 60
    hrs = mins // 60
    mins = mins % 60
    print(f'\nTotal runtime: {runtime:.2f}s = {int(hrs):02d}:' +
          f'{int(mins):02d}:{round(seconds)} h:m:s')
    ops['runtime'] = runtime 

    io.save_ops(ops, results_dir)

    return ops, similar_templates, is_ref, est_contam_rate, kept_spikes


def load_sorting(results_dir, device=None, load_extra_vars=False):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    results_dir = Path(results_dir)
    ops = io.load_ops(results_dir / 'ops.npy', device=device)
    similar_templates = np.load(results_dir / 'similar_templates.npy')

    clu = np.load(results_dir / 'spike_clusters.npy')
    st = np.load(results_dir / 'spike_times.npy')
    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    is_ref, est_contam_rate = CCG.refract(clu, st / ops['fs'],
                                          acg_threshold=acg_threshold,
                                          ccg_threshold=ccg_threshold)

    results = [ops, st, clu, similar_templates, is_ref, est_contam_rate]

    if load_extra_vars:
        # NOTE: tF and Wall always go on CPU, not CUDA
        tF = np.load(results_dir / 'tF.npy')
        tF = torch.from_numpy(tF)
        Wall = np.load(results_dir / 'Wall.npy')
        Wall = torch.from_numpy(Wall)
        full_st = np.load(results_dir / 'full_st.npy')
        full_clu = np.load(results_dir / 'full_clu.npy')
        full_amp = np.load(results_dir / 'full_amp.npy')
        results.extend([tF, Wall, full_st, full_clu, full_amp])

    return results
