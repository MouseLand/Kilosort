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

    Notes
    -----
    The returned settings dictionary contains the follow keys:

    n_chan_bin (required)
        Total number of channels in the binary file, which may be different
        from the number of channels containing ephys data. The value of this
        parameter *must* be specified by the user, or `run_kilosort` will
        raise a ValueError.
    fs
        Sampling frequency of probe.
    nt
        Number of samples per waveform. Also size of symmetric padding for filtering.
    spkTh
        Spike detection threshold for learned templates.
    Th_detect
        Spike detection threshold for universal templates.
    nskip
        Batch stride for computing whitening matrix.
    nt0min
        Sample index for aligning waveforms, so that their minimum value happens
        here. Default of 20 roughly aligns to the peak.
    NT
        Number of samples included in each batch of data.
    nblocks
        Number of non-overlapping blocks for drift correction (additional nblocks-1 
        blocks are created in the overlaps).
    binning_depth
        Vertical bin size in microns used for 2D histogram in drift correction. 
    sig_interp
        Sigma for interpolation (spatial standard deviation). Approximate smoothness scale
        for drift correction, in units of microns.
    probe_name
        Name of probe to load from Kilosort4's probe directory. This will only
        be used if no `probe` kwarg is specified for `run_kilosort`.
    tmin
        Time in seconds where data reference should begin. By default,
        begins at 0 seconds.
    tmax
        Time in seconds where data reference should end. By default, ends
        at the end of the recording.
    artifact_threshold
        If a batch contains absolute values above this number, it will be zeroed
        out under the assumption that a recording artifact is present.
        By default, the threshold is infinite (so that no zeroing occurs).
    whitening_range
        Number of nearby channels used to estimate the whitening matrix during
        preprocessing.
    dmin
        Vertical spacing of template centers used for spike detection,
        in microns.
    dminx
        Horizontal spacing of template centers used for spike detection,
        in microns.
    acg_threshold
        fraction of refractory period violations that are allowed in the ACG 
        compared to baseline; used to assign "good" units. 
    ccg_threshold
        fraction of refractory period violations that are allowed in the CCG
        compared to baseline; used to perform splits and merges during clustering. 
    cluster_downsampling
        inverse fraction of nodes used as landmarks during clustering (can be 1, but 
        that slows down the optimization). 
    cluster_pcs
        Maximum number of spatiotemporal PC features used for clustering.
    min_template_size
        Standard deviation of the smallest, spatial envelope Gaussian used for 
        universal templates
    template_sizes
        Number of sizes for universal spike templates (multiples of the min_template_size).
    nearest_chans
        Number of nearest channels to consider when finding local maxima
        during spike detection.
    nearest_templates
        Number of nearest spike template locations to consider when finding
        local maxima during spike detection.
    templates_from_data
        boolean flag indicating whether spike shapes used in universal templates should be 
        estimated from the data or loaded from the predefined templates
    n_templates
        Number of single-channel templates to use for the universal templates (only if 
        templates_from_data is True).
    n_pcs
        Number of single-channel PCs to use for extracting spike features (only if 
        templates_from_data is True).
    th_for_wPCA
        threshold for threshold crossings for estimating single-channel PCs and templates 
        
    """

    settings = {}
    settings['n_chan_bin']           = None   # Required, user must specify 
    settings['fs']                   = 30000
    settings['nt']                   = 61
    settings['Th']                   = 8
    settings['spkTh']                = 8
    settings['Th_detect']            = 9
    settings['nskip']                = 25
    settings['nt0min']               = int(20 * settings['nt']/61)
    settings['NT']                   = 2 * settings['fs']
    settings['nblocks']              = 5
    settings['binning_depth']        = 5
    settings['sig_interp']           = 20
    settings['probe_name']           = 'neuropixPhase3B1_kilosortChanMap.mat'
    settings['tmin']                 = 0.0
    settings['tmax']                 = np.inf
    settings['artifact_threshold']   = np.inf
    settings['whitening_range']      = 32
    settings['dmin']                 = None   # determine automatically
    settings['dminx']                = None   # determine automatically
    settings['acg_threshold']        = 0.2
    settings['ccg_threshold']        = 0.25
    settings['cluster_downsampling'] = 20
    settings['cluster_pcs']          = 64
    settings['min_template_size']    = 10
    settings['template_sizes']       = 5
    settings['nearest_chans']        = 10
    settings['nearest_templates']    = 100
    settings['templates_from_data']  = False
    settings['n_templates']          = 6
    settings['n_pcs']                = 6
    settings['th_for_wPCA']          = 6
    
    return settings


def run_kilosort(settings=None, probe=None, probe_name=None, data_dir=None,
                 filename=None, file_object=None, data_dtype=None,
                 results_dir=None, do_CAR=True, invert_sign=False,
                 device=torch.device('cuda'), progress_bar=None):
    """Spike sort the given dataset.
    
    Parameters
    ----------
    TODO

    file_object : array-like file object; optional.
        Must have 'shape' and 'dtype' attributes and support array-like
        indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
        array or memmap. Must specify a valid `filename` as well.
    
    Raises
    ------
    ValueError
        If settings[`n_chan_bin`] is None (default). User must specify, for
        example:  `run_kilosort(settings={'n_chan_bin': 385})`.

    """

    if not do_CAR:
        print("Skipping common average reference.")

    tic0 = time.time()

    # Configure settings, ops, and file paths
    d = default_settings()
    settings = {**d, **settings} if settings is not None else d
    if settings['n_chan_bin'] is None:
        raise ValueError(
            '`n_chan_bin` is a required setting. This is the total number of '
            'channels in the binary file, which may or may not be equal to the '
            'number of channels specified by the probe.'
            )
    # NOTE: Also modifies settings in-place
    filename, data_dir, results_dir, probe = \
        set_files(settings, filename, probe, probe_name, data_dir, results_dir)
    ops = initialize_ops(settings, probe, data_dtype, do_CAR, invert_sign)

    # Set preprocessing and drift correction parameters
    ops = compute_preprocessing(ops, device, tic0=tic0, file_object=file_object)
    np.random.seed(1)
    torch.cuda.manual_seed_all(1)
    torch.random.manual_seed(1)
    ops, bfile = compute_drift_correction(
        ops, device, tic0=tic0, progress_bar=progress_bar,
        file_object=file_object
        )
    
    # Save intermediate `ops` for use by GUI plots
    io.save_ops(ops, results_dir)

    # Sort spikes and save results
    st, clu, tF, Wall = sort_spikes(ops, device, bfile, tic0=tic0,
                                    progress_bar=progress_bar)
    ops, similar_templates, is_ref, est_contam_rate = \
        save_sorting(ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0)

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


def initialize_ops(settings, probe, data_dtype, do_CAR, invert_sign) -> dict:
    """Package settings and probe information into a single `ops` dictionary."""

    # TODO: Clean this up during refactor. Lots of confusing duplication here.
    ops = settings  
    ops['settings'] = settings 
    ops['probe'] = probe
    ops['data_dtype'] = data_dtype
    ops['do_CAR'] = do_CAR
    ops['invert_sign'] = invert_sign
    ops['NTbuff'] = ops['NT'] + 2 * ops['nt']
    ops['Nchan'] = len(probe['chanMap'])
    ops['n_chan_bin'] = settings['n_chan_bin']
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

    ops = datashift.run(ops, bfile, device=device, progress_bar=progress_bar)
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
    st0, tF, ops = spikedetect.run(ops, bfile, device=device, progress_bar=progress_bar)
    tF = torch.from_numpy(tF)
    print(f'{len(st0)} spikes extracted in {time.time()-tic : .2f}s; ' + 
            f'total {time.time()-tic0 : .2f}s')

    tic = time.time()
    print('\nFirst clustering')
    clu, Wall = clustering_qr.run(ops, st0, tF, mode='spikes', device=device,
                                  progress_bar=progress_bar)
    Wall3 = template_matching.postprocess_templates(Wall, ops, clu, st0, device=device)
    print(f'{clu.max()+1} clusters found, in {time.time()-tic : .2f}s; ' +
            f'total {time.time()-tic0 : .2f}s')
    
    tic = time.time()
    print('\nExtracting spikes using cluster waveforms')
    st, tF, tF2, ops = template_matching.extract(ops, bfile, Wall3, device=device,
                                                 progress_bar=progress_bar)
    print(f'{len(st)} spikes extracted in {time.time()-tic : .2f}s; ' +
            f'total {time.time()-tic0 : .2f}s')

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

    return st, clu, tF, Wall


def save_sorting(ops, results_dir, st, clu, tF, Wall, imin, tic0=np.nan):  
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
    results_dir, similar_templates, is_ref, est_contam_rate = io.save_to_phy(
            st, clu, tF, Wall, ops['probe'], ops, imin, results_dir=results_dir,
            data_dtype=ops['data_dtype']
            )
    print(f'{int(is_ref.sum())} units found with good refractory periods')
    
    runtime = time.time()-tic0
    print(f'\nTotal runtime: {runtime:.2f}s = {int(runtime//3600):02d}:' +
          f'{int(runtime//60):02d}:{int(runtime%60)} h:m:s')
    ops['runtime'] = runtime 

    io.save_ops(ops, results_dir)

    return ops, similar_templates, is_ref, est_contam_rate



