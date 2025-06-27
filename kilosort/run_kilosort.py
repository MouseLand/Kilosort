import time
from pathlib import Path
import logging
import warnings
import platform
logger = logging.getLogger(__name__)

import numpy as np
import torch

import kilosort
from kilosort import (preprocessing, datashift, template_matching, clustering_qr, 
                      clustering_qr, io, spikedetect, CCG, PROBE_DIR)
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.utils import (
    log_performance, log_cuda_details, probe_as_string, ops_as_string,
    get_performance, log_sorting_summary
    )
import kilosort.plots as kplots

RECOGNIZED_SETTINGS = list(DEFAULT_SETTINGS.keys())
RECOGNIZED_SETTINGS.extend([
    'filename', 'data_dir', 'results_dir', 'probe_name', 'probe_path',
])
# These get mixed in with the other parameters when running through the GUI.
# When using the API, these should NOT be included in a settings dictionary
# even if they share a name with run_kilosort options.
GUI_SETTINGS = [
    'data_file_path', 'probe', 'data_dtype', 'save_preprocessed_copy',
    'clear_cache', 'do_CAR', 'invert_sign', 'verbose_log'
]


def run_kilosort(settings, probe=None, probe_name=None, filename=None,
                 data_dir=None, file_object=None, results_dir=None,
                 data_dtype=None, do_CAR=True, invert_sign=False, device=None,
                 progress_bar=None, save_extra_vars=False, clear_cache=False,
                 save_preprocessed_copy=False, bad_channels=None, shank_idx=None,
                 verbose_console=False, verbose_log=False):
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
    filename: Path-like or list of Path-likes; optional.
        Full path to binary data file(s). If specified, will also set
        `data_dir = filename.parent`. If `filename` is a list, files will be
        treated as a single recording concatenated in time in the order provided.
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
    clear_cache : bool; default=False.
        If True, force pytorch to free up memory reserved for its cache in
        between memory-intensive operations.
        Note that setting `clear_cache=True` is NOT recommended unless you
        encounter GPU out-of-memory errors, since this can result in slower
        sorting.
    save_preprocessed_copy : bool; default=False.
        If True, save a pre-processed copy of the data (including drift
        correction) to `temp_wh.dat` in the results directory and format Phy
        output to use that copy of the data.
    bad_channels : list; optional.
        A list of channel indices (rows in the binary file) that should not be
        included in sorting. Listing channels here is equivalent to excluding
        them from the probe dictionary.
    shank_idx : float or list; optional.
        If not None, only channels from the specified shank index will be used.
        If a list is provided, each shank will be sorted sequentially and results
        will be saved in separate subfolders. Note that the shank_idx value(s)
        must match the actual value specified in `probe['kcoords']`. For example,
        `probe_idx=0` will not work if `probe['kcoords']` uses 1,2,3,4.
    verbose_console : bool; default=False.
        If True, set logging level for console output to `DEBUG` instead
        of `INFO`, so that additional information normally only saved to the
        log file will also show up in real time while sorting.
    verbose_log : bool; default=False.
        If True, include additional debug-level logging statements for some
        steps. This provides more detail for debugging, but may impact
        performance.
    
    Raises
    ------
    ValueError
        If settings[`n_chan_bin`] is None (default). User must specify, for
        example:  `run_kilosort(settings={'n_chan_bin': 385})`.

    Returns
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    st : np.ndarray
        3-column array of peak time (in samples), template, and thresold
        amplitude for each spike.
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st[:,0]`.
    tF : torch.Tensor
        PC features for each spike, with shape
        (n_spikes, nearest_chans, n_pcs)
    Wall : torch.Tensor
        PC feature representation of spike waveforms for each cluster, with shape
        (n_clusters, n_channels, n_pcs).
    similar_templates : np.ndarray.
        Similarity score between each pair of clusters, computed as correlation
        between clusters. Shape (n_clusters, n_clusters).
    is_ref : np.ndarray.
        1D boolean array with shape (n_clusters,) indicating whether each
        cluster is refractory.
    est_contam_rate : np.ndarray.
        Contamination rate for each cluster, computed as fraction of refractory
        period violations relative to expectation based on a Poisson process.
        Shape (n_clusters,).
    kept_spikes : np.ndarray.
        Boolean mask with shape (n_spikes,) that is False for spikes that were
        removed by `kilosort.postprocessing.remove_duplicate_spikes`
        and True otherwise.

    Notes
    -----
    For documentation of saved files, see `kilosort.io.save_to_phy`.

    """

    # Configure settings, ops, and file paths
    if settings is None or settings.get('n_chan_bin', None) is None:
        raise ValueError(
            '`n_chan_bin` is a required setting. This is the total number of '
            'channels in the binary file, which may or may not be equal to the '
            'number of channels specified by the probe.'
            )
    settings = {**DEFAULT_SETTINGS, **settings}
    # NOTE: This modifies settings in-place
    if not isinstance(shank_idx, list): shank_idx = [shank_idx]
    for idx in shank_idx:
        _filename, _data_dir, _results_dir, _probe = \
            set_files(settings, filename, probe, probe_name, data_dir,
                      results_dir, bad_channels, idx)
        setup_logger(_results_dir, verbose_console=verbose_console)

        ops, st, clu, tF, Wall, similar_templates, \
            is_ref, est_contam_rate, kept_spikes = _sort(
                _filename, _results_dir, _probe, settings, data_dtype, device,
                do_CAR, clear_cache, invert_sign, save_preprocessed_copy,
                verbose_log, save_extra_vars, file_object, progress_bar,
            )

    return ops, st, clu, tF, Wall, similar_templates, \
           is_ref, est_contam_rate, kept_spikes


def _sort(filename, results_dir, probe, settings, data_dtype, device, do_CAR,
          clear_cache, invert_sign, save_preprocessed_copy, verbose_log,
          save_extra_vars, file_object, progress_bar, gui_sorter=None):
    """Run sorting pipeline. See `run_kilosort` for documentation.
    
    Notes
    -----
    filename is expected to be a list of Paths at this point, even if it's
    a singleton list.
    
    """

    try:
        logger.info(f"Kilosort version {kilosort.__version__}")
        logger.info(f"Python version {platform.python_version()}")
        logger.info('-'*40)

        logger.info('System information:')
        logger.info(f'{platform.platform()} {platform.machine()}')
        logger.info(platform.processor())
        if device is None:
            if torch.cuda.is_available():
                logger.info('Using GPU for PyTorch computations. '
                            'Specify `device` to change this.')
                device = torch.device('cuda')
            else:
                logger.info('Using CPU for PyTorch computations. '
                            'Specify `device` to change this.')
                device = torch.device('cpu')

        if device != torch.device('cpu'):
            memory = torch.cuda.get_device_properties(device).total_memory/1024**3
            logger.info(f'Using CUDA device: {torch.cuda.get_device_name()} {memory:.2f}GB')

        logger.info('-'*40)
        if len(filename) == 1:
            logger.info(f"Sorting {filename}")
        else:
            logger.info(f"Sorting {filename[0].parent}/... (multiple files)")

        if data_dtype is None:
            logger.info(
                "Interpreting binary file as default dtype='int16'. If data was "
                "saved in a different format, specify `data_dtype`."
                )
            data_dtype = 'int16'

        if not do_CAR:
            logger.info("Skipping common average reference.")
        if clear_cache:
            logger.info('clear_cache=True')

        if probe['chanMap'].max() >= settings['n_chan_bin']:
            raise ValueError(
                f'Largest value of chanMap exceeds channel count of data, '
                'make sure chanMap is 0-indexed.'
            )

        tic0 = time.time()
        ops, settings = initialize_ops(
            settings, probe, data_dtype, do_CAR, invert_sign,
            device, save_preprocessed_copy, gui_mode=(gui_sorter is not None)
            )
        
        # Pretty-print ops and probe for log
        logger.debug(f"Initial ops:\n\n{ops_as_string(ops)}\n")
        logger.debug(f"Probe dictionary:\n\n{probe_as_string(ops['probe'])}\n")

        # Baseline performance metrics
        log_performance(logger, 'info', 'Resource usage before sorting')

        # Set preprocessing and drift correction parameters
        ops = compute_preprocessing(ops, device, tic0=tic0, file_object=file_object)
        np.random.seed(1)
        torch.cuda.manual_seed_all(1)
        torch.random.manual_seed(1) 
        ops, bfile, st0 = compute_drift_correction(
            ops, device, tic0=tic0, progress_bar=progress_bar,
            file_object=file_object, clear_cache=clear_cache,
            verbose=verbose_log
            )

        # Save preprocessing steps
        if save_preprocessed_copy:
            io.save_preprocessing(results_dir / 'temp_wh.dat', ops, bfile)
            log_performance(logger, 'info', 'Resource usage after saving preprocessing.',
                            reset=True)

        logger.info('Generating drift plots ...')
        # st0 will be None if nblocks = 0 (no drift correction)
        if st0 is not None:
            if gui_sorter is not None:
                gui_sorter.dshift = ops['dshift']
                gui_sorter.st0 = st0
                gui_sorter.plotDataReady.emit('drift')
            else:
                kplots.plot_drift_amount(ops, results_dir)
                kplots.plot_drift_scatter(st0, results_dir)

        # Sort spikes and save results
        st,tF, Wall0, clu0 = detect_spikes(
            ops, device, bfile, tic0=tic0, progress_bar=progress_bar,
            clear_cache=clear_cache, verbose=verbose_log
            )

        logger.info('Generating diagnostic plots ...')
        if gui_sorter is not None:
            gui_sorter.Wall0 = Wall0
            gui_sorter.wPCA = torch.clone(ops['wPCA'].cpu()).numpy()
            gui_sorter.clu0 = clu0
            gui_sorter.plotDataReady.emit('diagnostics')
        else:
            kplots.plot_diagnostics(Wall0, clu0, ops, results_dir)

        clu, Wall, st, tF = cluster_spikes(
            st, tF, ops, device, bfile, tic0=tic0, progress_bar=progress_bar,
            clear_cache=clear_cache, verbose=verbose_log,
            )
        ops, similar_templates, is_ref, est_contam_rate, kept_spikes = \
            save_sorting(
                ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0,
                save_extra_vars=save_extra_vars,
                save_preprocessed_copy=save_preprocessed_copy,
                skip_dat_path=(file_object is not None)
                )
        if torch.cuda.is_available():
            ops['cuda_postproc'] = torch.cuda.memory_stats(device)

        logger.info('Generating spike position plot ...')
        if gui_sorter is not None:
            gui_sorter.clu = clu[kept_spikes]
            gui_sorter.is_refractory = is_ref
            gui_sorter.plotDataReady.emit('probe')
        else:
            kplots.plot_spike_positions(clu[kept_spikes], is_ref, results_dir)
        logger.info('Sorting finished.')
        log_sorting_summary(ops, log=logger, level='info')
        
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            logger.exception('Out of memory error, printing performance...')
            log_cuda_details(logger)
            log_performance(logger, level='info')

        # This makes sure the full traceback is written to log file.
        logger.exception('Encountered error in `run_kilosort`:')
        # Annoyingly, this will print the error message twice for console, but
        # I haven't found a good way around that.
        raise
    
    finally:
        close_logger()


    return ops, st, clu, tF, Wall, similar_templates, \
           is_ref, est_contam_rate, kept_spikes


def set_files(settings, filename, probe, probe_name, data_dir, results_dir,
              bad_channels, shank_idx):
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
        filename = [filename]
    else:
        if not isinstance(filename, list):
            filename = [filename]
        filename = [Path(f) for f in filename]
        for f in filename:
            if not f.exists():
                raise FileExistsError(f"filename '{filename}' does not exist")
        data_dir = filename[0].parent
        
    # Convert paths to strings when saving to ops, otherwise ops can only
    # be loaded on the operating system that originally ran the code.
    settings['filename'] = filename
    settings['data_dir'] = data_dir

    # Try to set results_dir based on settings, otherwise use default.
    results_dir = settings.get('results_dir', None) if results_dir is None else results_dir
    results_dir = Path(results_dir).resolve() if results_dir is not None else None
    if results_dir is None:
        results_dir = data_dir / 'kilosort4'
    if shank_idx is not None:
        results_dir = results_dir / f'shank_{shank_idx}'
    # Make sure results directory exists
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # find probe configuration file and load
    if probe is None:
        if probe_name is not None:     probe_path = PROBE_DIR / probe_name
        elif 'probe_name' in settings: probe_path = PROBE_DIR / settings['probe_name']
        elif 'probe_path' in settings: probe_path = Path(settings['probe_path']).resolve()
        else: raise ValueError('no probe_name or probe_path provided, set probe_name=')
        if not probe_path.exists():
            raise FileExistsError(f"probe_path '{probe_path}' does not exist")
        
        probe  = io.load_probe(probe_path)
    else:
        # Make sure xc, yc are float32, otherwise there are casting problems
        # with some pytorch functions.
        probe['xc'] = probe['xc'].astype(np.float32)
        probe['yc'] = probe['yc'].astype(np.float32)

    # Let user know if there are too many dimensions in probe entries.
    # Don't want to automatically flatten them incase they've made assumptions
    # about higher-D ordering.
    for k in ['xc', 'yc', 'kcoords', 'chanMap']:
        if probe[k].ndim > 1:
            raise ValueError(f"Array-valued probe entries should have 1 dim, "
                             f"but key: {k} has ndim == {probe[k].ndim}.")

    if bad_channels is not None:
        probe = io.remove_bad_channels(probe, bad_channels)
    if shank_idx is not None:
        probe = io.select_shank(probe, shank_idx)

    return filename, data_dir, results_dir, probe


def setup_logger(results_dir, verbose_console=False):
    results_dir = Path(results_dir)
    
    # Get root logger for Kilosort application
    ks_log = logging.getLogger('kilosort')
    ks_log.setLevel(logging.DEBUG)

    # Add file handler at debug level, include timestamps and logging level
    # in text output.
    file = logging.FileHandler(results_dir / 'kilosort4.log', mode='w')
    file.setLevel(logging.DEBUG)
    text_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    file_formatter = logging.Formatter(text_format)
    file.setFormatter(file_formatter)

    # Skip this if the handlers were already added, like when running multiple
    # times in a single session.
    if not ks_log.handlers:
        # Add console handler at info level with shorter messages,
        # unless verbose is requested.
        console = logging.StreamHandler()
        if verbose_console:
            console.setLevel(logging.DEBUG)
            console.setFormatter(file_formatter)
        else:
            console.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(name)-12s: %(message)s')
            console.setFormatter(console_formatter)
        ks_log.addHandler(console)

    # Always add file handler since log file might change locations
    ks_log.addHandler(file)


def close_logger():
    ks_log = logging.getLogger('kilosort')
    for handler in ks_log.handlers.copy():
        ks_log.removeHandler(handler)
        handler.close()


def initialize_ops(settings, probe, data_dtype, do_CAR, invert_sign,
                   device, save_preprocessed_copy, gui_mode=False) -> dict:
    """Package settings and probe information into a single `ops` dictionary."""

    settings = settings.copy()
    if settings['nt0min'] is None:
        settings['nt0min'] = int(20 * settings['nt']/61)
    if settings['max_channel_distance'] is None:
        # Default used to be None, now it's a constant. Adding this so that
        # cached settings values in the GUI don't cause disruption.
        settings['max_channel_distance'] = DEFAULT_SETTINGS['max_channel_distance']

    if settings['nearest_chans'] > len(probe['chanMap']):
        msg = f"""
            Parameter `nearest_chans` must be less than or equal to the number 
            of data channels being sorted.\n
            Changing from {settings['nearest_chans']} to {len(probe['chanMap'])}.
            """
        warnings.warn(msg, UserWarning)
        settings['nearest_chans'] = len(probe['chanMap'])

    if 'duplicate_spike_bins' in settings:
        msg = """
            The `duplicate_spike_bins` parameter has been replaced with 
            `duplicate_spike_ms`. Specifying the former will have no effect, 
            since it gets overwritten based on sampling rate.
            """
        warnings.warn(msg, DeprecationWarning)
    dup_bins = int(settings['duplicate_spike_ms'] * (settings['fs']/1000))

    # If running through GUI, also allow some additional relevant keys in
    # settings dictionary.
    recognized = RECOGNIZED_SETTINGS.copy()
    if gui_mode:
        recognized.extend(GUI_SETTINGS.copy())

    # Raise an error if there are unrecognized settings entries to make users
    # aware if they've made a typo, are using a deprecated setting, etc.
    unrecognized = []
    for k, _ in settings.items():
        if k not in recognized:
            unrecognized.append(k)
    if len(unrecognized) > 0:
        logger.info('Unrecognized keys found in `settings`')
        logger.info('See `kilosort.run_kilosort.RECOGNIZED_SETTINGS`')
        raise ValueError(f'Unrecognized settings: {unrecognized}')


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
    ops['duplicate_spike_bins'] = dup_bins
    ops['torch_device'] = str(device)
    ops['save_preprocessed_copy'] = save_preprocessed_copy

    if not settings['templates_from_data'] and settings['nt'] != 61:
        raise ValueError('If using pre-computed universal templates '
                         '(templates_from_data=False), nt must be 61')

    ops = {**ops, **probe}

    return ops, settings

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
        ops['settings']['artifact_threshold'],
        ops['settings']['shift'],
        ops['settings']['scale']
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
    logger.info(' ')
    logger.info('Computing preprocessing variables.')
    logger.info('-'*40)

    n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
        xc, yc, tmin, tmax, artifact, shift, scale = get_run_parameters(ops)
    nskip = ops['settings']['nskip']
    whitening_range = ops['settings']['whitening_range']
    
    # Compute high pass filter
    cutoff = ops['settings']['highpass_cutoff']
    hp_filter = preprocessing.get_highpass_filter(fs, cutoff, device=device)
    # Compute whitening matrix
    bfile = io.BinaryFiltered(ops['filename'], n_chan_bin, fs, NT, nt, twav_min,
                              chan_map, hp_filter, device=device, do_CAR=do_CAR,
                              invert_sign=invert, dtype=dtype, tmin=tmin,
                              tmax=tmax, artifact_threshold=artifact,
                              shift=shift, scale=scale, file_object=file_object)

    logger.info(f'N samples: {bfile.n_samples}')
    logger.info(f'N seconds: {bfile.n_samples/fs}')
    logger.info(f'N batches: {bfile.n_batches}')

    whiten_mat = preprocessing.get_whitening_matrix(bfile, xc, yc, nskip=nskip,
                                                    nrange=whitening_range)


    # Save results
    ops['Nbatches'] = bfile.n_batches
    ops['preprocessing'] = {}
    ops['preprocessing']['whiten_mat'] = whiten_mat
    ops['preprocessing']['hp_filter'] = hp_filter
    ops['Wrot'] = whiten_mat
    ops['fwav'] = hp_filter

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_preproc'] = elapsed
    ops['usage_preproc'] = get_performance()
    logger.info(f'Preprocessing filters computed in {elapsed:.2f}s; ' +
                f'total {total:.2f}s')
    logger.debug(f'hp_filter shape: {hp_filter.shape}')
    logger.debug(f'whiten_mat shape: {whiten_mat.shape}')
    # Check scale of data for log file
    b1 = bfile.padded_batch_to_torch(0).cpu().numpy()
    logger.debug(f"First batch min, max: {b1.min(), b1.max()}")

    log_performance(logger, 'info', 'Resource usage after preprocessing',
                    reset=True)

    return ops


def compute_drift_correction(ops, device, tic0=np.nan, progress_bar=None,
                             file_object=None, clear_cache=False, verbose=False):
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
    clear_cache : bool; False.
        If True, force pytorch to clear cached cuda memory after some
        memory-intensive steps in the pipeline.
    verbose : bool; False.
        If true, include additional debug-level logging statements.

    Returns
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    bfile : kilosort.io.BinaryFiltered
        Wrapped file object for handling data.
    st0 : np.ndarray.
        Intermediate spike times variable with 6 columns. This is only used
        for generating the 'Drift Scatter' plot through the GUI.
    
    """

    tic = time.time()
    logger.info(' ')
    logger.info('Computing drift correction.')
    logger.info('-'*40)

    n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
        _, _, tmin, tmax, artifact, shift, scale = get_run_parameters(ops)
    hp_filter = ops['preprocessing']['hp_filter']
    whiten_mat = ops['preprocessing']['whiten_mat']
    bfile = io.BinaryFiltered(
        ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
        hp_filter=hp_filter, whiten_mat=whiten_mat, device=device, do_CAR=do_CAR,
        invert_sign=invert, dtype=dtype, tmin=tmin, tmax=tmax,
        artifact_threshold=artifact, shift=shift, scale=scale,
        file_object=file_object
        )

    ops, st = datashift.run(ops, bfile, device=device, progress_bar=progress_bar,
                            clear_cache=clear_cache, verbose=verbose)
    
    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_drift'] = elapsed
    ops['usage_drift'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_drift'] = torch.cuda.memory_stats()
    logger.info(f'drift computed in {elapsed:.2f}s; total {total:.2f}s')

    if st is not None:
        logger.debug(f'st shape: {st.shape}')
        logger.debug(f'yblk shape: {ops["yblk"].shape}')
        logger.debug(f'dshift shape: {ops["dshift"].shape}')
        logger.debug(f'iKxx shape: {ops["iKxx"].shape}')
    
    # binary file with drift correction
    bfile = io.BinaryFiltered(
        ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
        hp_filter=hp_filter, whiten_mat=whiten_mat, device=device,
        dshift=ops['dshift'], do_CAR=do_CAR, dtype=dtype, tmin=tmin, tmax=tmax,
        artifact_threshold=artifact, shift=shift, scale=scale,
        file_object=file_object
        )

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after drift correction',
                    reset=True)

    return ops, bfile, st


def detect_spikes(ops, device, bfile, tic0=np.nan, progress_bar=None,
                  clear_cache=False, verbose=False):
    """Detect spikes via template deconvolution.
    
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
    clear_cache : bool; False.
        If True, force pytorch to clear cached cuda memory after some
        memory-intensive steps in the pipeline.
    verbose : bool; False.
        If true, include additional debug-level logging statements.

    Returns
    -------
    st : np.ndarray
        3-column array of peak time (in samples), template, and thresold
        amplitude for each spike.
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st`.
    tF : torch.Tensor
        PC features for each spike, with shape
        (n_spikes, nearest_chans, n_pcs)
    Wall : torch.Tensor
        PC feature representation of spike waveforms for each cluster, with shape
        (n_clusters, n_channels, n_pcs).

    """

    tic = time.time()
    logger.info(' ')
    logger.info(f'Extracting spikes using templates')
    logger.info('-'*40)
    st0, tF, ops = spikedetect.run(
        ops, bfile, device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )
    tF = torch.from_numpy(tF)

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_st0'] = elapsed
    ops['usage_st0'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_st0'] = torch.cuda.memory_stats(device)
    logger.info(f'{len(st0)} spikes extracted in {elapsed:.2f}s; ' + 
                f'total {total:.2f}s')
    logger.debug(f'st0 shape: {st0.shape}')
    logger.debug(f'tF shape: {tF.shape}')
    if len(st0) == 0:
        raise ValueError('No spikes detected, cannot continue sorting.')
    log_performance(logger, 'info', 'Resource usage after spike detect (univ)',
                    reset=True)

    tic = time.time()
    logger.info(' ')
    logger.info('First clustering')
    logger.info('-'*40)
    clu, Wall = clustering_qr.run(
        ops, st0, tF, mode='spikes', device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )
    Wall3 = template_matching.postprocess_templates(
        Wall, ops, clu, st0, tF, device=device
        )

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_clu0'] = elapsed
    ops['usage_clu0'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_clu0'] = torch.cuda.memory_stats(device)
    logger.info(f'{clu.max()+1} clusters found, in {elapsed:.2f}s; ' +
                f'total {total:.2f}s')
    logger.debug(f'clu shape: {clu.shape}')
    logger.debug(f'Wall shape: {Wall.shape}')
    log_performance(logger, 'info', 'Resource usage after first clustering',
                    reset=True)
    
    tic = time.time()
    logger.info(' ')
    logger.info('Extracting spikes using cluster waveforms')
    logger.info('-'*40)
    st, tF, ops = template_matching.extract(
        ops, bfile, Wall3, device=device, progress_bar=progress_bar
        )
    
    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_st'] = elapsed
    ops['usage_st'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_st'] = torch.cuda.memory_stats(device)
    logger.info(f'{len(st)} spikes extracted in {elapsed:.2f}s; ' +
                f'total {total:.2f}s')
    logger.debug(f'st shape: {st.shape}')
    logger.debug(f'tF shape: {tF.shape}')
    logger.debug(f'iCC shape: {ops["iCC"].shape}')
    logger.debug(f'iU shape: {ops["iU"].shape}')

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after spike detect (learned)',
                    reset=True)

    return st, tF, Wall, clu


def cluster_spikes(st, tF, ops, device, bfile, tic0=np.nan, progress_bar=None,
                   clear_cache=False, verbose=False):
    """Cluster spikes using graph-based methods.
    
    Parameters
    ----------
    st : np.ndarray
        3-column array of peak time (in samples), template, and thresold
        amplitude for each spike.
    tF : torch.Tensor
        PC features for each spike, with shape
        (n_spikes, nearest_chans, n_pcs)
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
    clear_cache : bool; False.
        If True, force pytorch to clear cached cuda memory after some
        memory-intensive steps in the pipeline.
    verbose : bool; False.
        If True, include additional debug-level logging statements.

    Returns
    -------
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st`.
    Wall : torch.Tensor
        PC feature representation of spike waveforms for each cluster, with shape
        (n_clusters, n_channels, n_pcs).
    
    """
    tic = time.time()
    logger.info(' ')
    logger.info('Final clustering')
    logger.info('-'*40)
    clu, Wall = clustering_qr.run(
        ops, st, tF,  mode = 'template', device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )
    
    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_clu'] = elapsed
    ops['usage_clu'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_clu'] = torch.cuda.memory_stats(device)
    logger.info(f'{clu.max()+1} clusters found, in {elapsed:.2f}s; ' + 
                f'total {total:.2f}s')
    logger.debug(f'clu shape: {clu.shape}')
    logger.debug(f'Wall shape: {Wall.shape}')

    tic = time.time()
    logger.info(' ')
    logger.info('Merging clusters')
    logger.info('-'*40)
    Wall, clu, is_ref, st, tF = template_matching.merging_function(
        ops, Wall, clu, st, tF, device=device, check_dt=True
        )
    clu = clu.astype('int32')

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_merge'] = elapsed
    ops['usage_merge'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_merge'] = torch.cuda.memory_stats(device)
    logger.info(f'{clu.max()+1} units found, in {elapsed:.2f}s; ' + 
                f'total {total:.2f}s')
    logger.debug(f'clu shape: {clu.shape}')
    logger.debug(f'Wall shape: {Wall.shape}')

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after clustering',
                    reset=True)

    return clu, Wall, st, tF


def save_sorting(ops, results_dir, st, clu, tF, Wall, imin, tic0=np.nan,
                 save_extra_vars=False, save_preprocessed_copy=False,
                 skip_dat_path=False):  
    """Save sorting results, and format them for use with Phy

    Parameters
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    results_dir : pathlib.Path
        Directory where results should be saved.
    st : np.ndarray
        3-column array of peak time (in samples), template, and thresold
        amplitude for each spike.
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st[:,0]`.
    tF : torch.Tensor
        PC features for each spike, with shape
        (n_spikes, nearest_chans, n_pcs)
    Wall : torch.Tensor
        PC feature representation of spike waveforms for each cluster, with shape
        (n_clusters, n_channels, n_pcs).
    imin : int
        Minimum sample index used by BinaryRWFile, exported spike times will
        be shifted forward by this number.
    tic0 : float; default=np.nan.
        Start time of `run_kilosort`.
    save_extra_vars : bool; default=False.
        If True, save tF and Wall to disk along with copies of st, clu and
        amplitudes with no postprocessing applied.
    save_preprocessed_copy : bool; default=False.
        If True, save a pre-processed copy of the data (including drift
        correction) to `temp_wh.dat` in the results directory and format Phy
        output to use that copy of the data.
    skip_dat_path : bool; default=False.
        If True, will save `dat_path = 'no_path.bin'` in `params.py` in place
        of a real filename. This is done to prevent an error in Phy when filename
        has an unexpected format, like when using a `file_object` loaded from
        an external data format through SpikeInterface. The full filename(s) will
        still be included in `params.py` for reference, but will be commented out.

    Returns
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    similar_templates : np.ndarray.
        Similarity score between each pair of clusters, computed as correlation
        between clusters. Shape (n_clusters, n_clusters).
    is_ref : np.ndarray.
        1D boolean array with shape (n_clusters,) indicating whether each
        cluster is refractory.
    est_contam_rate : np.ndarray.
        Contamination rate for each cluster, computed as fraction of refractory
        period violations relative to expectation based on a Poisson process.
        Shape (n_clusters,).
    kept_spikes : np.ndarray.
        Boolean mask with shape (n_spikes,) that is False for spikes that were
        removed by `kilosort.postprocessing.remove_duplicate_spikes`
        and True otherwise.

    Notes
    -----
    For documentation of saved files, see `kilosort.io.save_to_phy`.

    """

    tic = time.time()
    logger.info(' ')
    logger.info('Saving to phy and computing refractory periods')
    logger.info('-'*40)
    results_dir, similar_templates, is_ref, est_contam_rate, kept_spikes = \
        io.save_to_phy(
            st, clu, tF, Wall, ops['probe'], ops, imin, results_dir=results_dir,
            data_dtype=ops['data_dtype'], save_extra_vars=save_extra_vars,
            save_preprocessed_copy=save_preprocessed_copy,
            skip_dat_path=skip_dat_path
            )
    logger.info(f'{int(is_ref.sum())} units found with good refractory periods')
    
    ops['n_units_total'] = np.unique(clu).size
    ops['n_units_good'] = int(is_ref.sum())
    ops['n_spikes'] = st[kept_spikes].shape[0]
    if ops.get('dshift', None) is not None:
        ops['mean_drift'] = np.abs(ops['dshift']).mean(axis=0)[0]
    else:
        ops['mean_drift'] = np.nan

    elapsed = elapsed = time.time() - tic
    ops['runtime_postproc'] = elapsed
    ops['usage_postproc'] = get_performance()
    logger.info(f'Exporting to Phy took: {elapsed:.2f}s')

    runtime = time.time()-tic0
    seconds = runtime % 60
    mins = runtime // 60
    hrs = mins // 60
    mins = mins % 60

    logger.info(f'Total runtime: {runtime:.2f}s = {int(hrs):02d}:' +
                f'{int(mins):02d}:{round(seconds)} h:m:s')
    ops['runtime'] = runtime 
    io.save_ops(ops, results_dir)
    logger.info(f'Sorting output saved in: {results_dir}.')

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after saving',
                    reset=True)

    return ops, similar_templates, is_ref, est_contam_rate, kept_spikes


def load_sorting(results_dir, device=None, load_extra_vars=False):
    '''Load saved sorting results into memory.
    
    Parameters
    ----------
    results_dir : str or pathlib.Path
        Directory where results were saved.
    device : torch.device; optional.
        CPU or GPU device to use to load Pytorch tensors. By default, PyTorch
        will use the first detected GPU. If no GPUs are detected, CPU will be
        used. To set this manually, specify `device = torch.device(<device_name>)`.
        See PyTorch documentation for full description.
    load_extra_vars : default=False.
        If True, load tF, Wall, and full copies of st, clu, and spike amplitudes
        in addition to the other variables.

    Returns
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    st : np.ndarray
        1D vector of spike times (in samples) for all clusters. This is *only*
        the first column of the 3-column array returned by `run_kilosort`.
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st`.
    similar_templates : np.ndarray.
        Similarity score between each pair of clusters, computed as correlation
        between clusters. Shape (n_clusters, n_clusters).
    is_ref : np.ndarray.
        1D boolean array with shape (n_clusters,) indicating whether each
        cluster is refractory.
    est_contam_rate : np.ndarray.
        Contamination rate for each cluster, computed as fraction of refractory
        period violations relative to expectation based on a Poisson process.
        Shape (n_clusters,).
    kept_spikes : np.ndarray.
        Boolean mask with shape (n_spikes,) that is False for spikes that were
        removed by `kilosort.postprocessing.remove_duplicate_spikes`
        and True otherwise.
    tF : torch.Tensor.
        Only returned if `load_extra_vars` is True.
        PC features for each spike, with shape (n_spikes, nearest_chans, n_pcs)
    Wall : torch.Tensor.
        Only returned if `load_extra_vars` is True.
        PC feature representation of spike waveforms for each cluster, with shape
        (n_clusters, n_channels, n_pcs).
    full_st : np.ndarray.
        Only returned if `load_extra_vars` is True.
        3-column array of peak time (in samples), template, and threshold amplitude for
        each spike.
        Includes spikes removed by `kilosort.postprocessing.remove_duplicate_spikes`.
    full_clu : np.ndarray.
        Only returned if `load_extra_vars` is True.
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st[:,0]`.
        Includes spikes removed by `kilosort.postprocessing.remove_duplicate_spikes`.
    full_amp : np.ndarray.
        Only returned if `load_extra_vars` is True.
        Per-spike amplitudes, computed as the L2 norm of the PC features
        for each spike.
        Includes spikes removed by `kilosort.postprocessing.remove_duplicate_spikes`.
    
    '''
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
    kept_spikes = np.load(results_dir / 'kept_spikes.npy')
    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    is_ref, est_contam_rate = CCG.refract(clu, st / ops['fs'],
                                          acg_threshold=acg_threshold,
                                          ccg_threshold=ccg_threshold)

    results = [ops, st, clu, similar_templates, is_ref,
               est_contam_rate, kept_spikes]

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
