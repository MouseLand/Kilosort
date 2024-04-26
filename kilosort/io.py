import json
from pathlib import Path
from typing import Tuple, Union
import os, shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
import time

from scipy.io import loadmat
import numpy as np
import torch
from torch.fft import fft, ifft, fftshift

from kilosort import CCG
from kilosort.preprocessing import get_drift_matrix, fft_highpass
from kilosort.postprocessing import (
    remove_duplicates, compute_spike_positions, make_pc_features
    )

_torch_warning = ".*PyTorch does not support non-writable tensors"


def find_binary(data_dir: Union[str, os.PathLike]) -> Path:
    """Find binary file in `data_dir`."""

    data_dir = Path(data_dir)
    filenames = list(data_dir.glob('*.bin')) + list(data_dir.glob('*.bat')) \
                + list(data_dir.glob('*.dat')) + list(data_dir.glob('*.raw'))
    if len(filenames) == 0:
        raise FileNotFoundError(
            'No binary file found in folder. Expected extensions are:\n'
            '*.bin, *.bat, *.dat, or *.raw.'
            )

    # TODO: Why give this preference? Not all binary files will have this tag.
    # If there are multiple binary files, find one with "ap" tag
    if len(filenames) > 1:
        filenames = [f for f in filenames if 'ap' in f.as_posix()]

    # If there is still more than one, raise an error, user needs to specify
    # full path.
    if len(filenames) > 1:
        raise ValueError('Multiple binary files in folder with "ap" tag, '
                         'please specify filename')

    return filenames[0]


def load_probe(probe_path):
    """Load a .mat probe file from Kilosort2, or a PRB file and returns a dictionary
    
    adapted from https://github.com/MouseLand/pykilosort/blob/5712cfd2722a20554fa5077dd8699f68508d1b1a/pykilosort/utils.py#L592

    """
    probe = {}
    probe_path = Path(probe_path).resolve()
    required_keys = ['chanMap', 'yc', 'xc', 'n_chan']

    if probe_path.suffix == '.prb':
        # Support for PRB files.
        # !DOES NOT WORK FOR PHASE3A PROBES WITH DISCONNECTED CHANNELS!
        # Also does not remove reference channel in PHASE3B probes
        contents = probe_path.read_text()
        metadata = {}
        exec(contents, {}, metadata)
        probe['chanMap'] = []
        probe['xc'] = []
        probe['yc'] = []
        probe['kcoords'] = []
        probe['n_chan'] = 0 
        for cg in sorted(metadata['channel_groups']):
            d = metadata['channel_groups'][cg]
            ch = d['channels']
            pos = d.get('geometry', {})
            probe['chanMap'].append(ch)
            probe['n_chan'] += len(ch)
            probe['xc'].append([pos[c][0] for c in ch])
            probe['yc'].append([pos[c][1] for c in ch])
            probe['kcoords'].append([cg for c in ch])
        probe['chanMap'] = np.concatenate(probe['chanMap']).ravel().astype(np.int32)
        probe['xc'] = np.concatenate(probe['xc']).astype('float32')
        probe['yc'] = np.concatenate(probe['yc']).astype('float32')
        probe['kcoords'] = np.concatenate(probe['kcoords']).astype('float32')

    elif probe_path.suffix == '.mat':
        mat = loadmat(probe_path)
        connected = mat['connected'].ravel().astype('bool')
        probe['xc'] = mat['xcoords'].ravel().astype(np.float32)[connected]
        nc = len(probe['xc'])
        probe['yc'] = mat['ycoords'].ravel().astype(np.float32)[connected]
        probe['kcoords'] = mat.get('kcoords', np.zeros(nc)).ravel().astype(np.float32)
        probe['chanMap'] = (mat['chanMap'] - 1).ravel().astype(np.int32)[connected]  # NOTE: 0-indexing in Python
        probe['n_chan'] = (mat['chanMap'] - 1).ravel().astype(np.int32).shape[0]  # NOTE: should match the # of columns in the raw data

    elif probe_path.suffix == '.json':
        with open(probe_path, 'r') as f:
            probe = json.load(f)
        for k in list(probe.keys()):
            # Convert lists back to arrays
            v = probe[k]
            if isinstance(v, list):
                dtype = np.int32 if k == 'chanMap' else np.float32
                probe[k] = np.array(v, dtype=dtype)

    for n in required_keys:
        assert n in probe.keys()

    return probe

  
def save_probe(probe_dict, filepath):
    """Save a probe dictionary to a .json text file.

    Parameters
    ----------
    probe_dict : dict
        A dictionary containing probe information in the format expected by
        Kilosort4, with keys 'chanMap', 'xc', 'yc', and 'kcoords'.
    filepath : str or pathlib.Path
        Location where .json file should be stored.

    Raises
    ------
    RuntimeWarning
        If filepath does not end in '.json'
    
    """

    if Path(filepath).suffix != '.json':
        raise RuntimeWarning(
            'Saving json probe to a file whose suffix is not .json. '
            'kilosort.io.load_probe will not recognize this file.' 
        )

    d = probe_dict.copy()
    # Convert arrays to lists, since arrays aren't json-able
    for k in list(d.keys()):
        v = d[k]
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
    
    with open(filepath, 'w') as f:
        f.write(json.dumps(d))


def save_to_phy(st, clu, tF, Wall, probe, ops, imin, results_dir=None,
                data_dtype=None, save_extra_vars=False):

    if results_dir is None:
        results_dir = ops['data_dir'].joinpath('kilosort4')
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    # probe properties
    chan_map = probe['chanMap']
    channel_positions = np.stack((probe['xc'], probe['yc']), axis=-1)
    np.save((results_dir / 'channel_map.npy'), chan_map)
    np.save((results_dir / 'channel_positions.npy'), channel_positions)

    # whitening matrix ** saving real whitening matrix doesn't work with phy currently
    whitening_mat = ops['Wrot'].cpu().numpy()
    np.save((results_dir / 'whitening_mat_dat.npy'), whitening_mat)
    whitening_mat = 0.005 * np.eye(len(chan_map), dtype='float32')
    whitening_mat_inv = np.linalg.inv(whitening_mat + 1e-5 * np.eye(whitening_mat.shape[0]))
    np.save((results_dir / 'whitening_mat.npy'), whitening_mat)
    np.save((results_dir / 'whitening_mat_inv.npy'), whitening_mat_inv)

    # spike properties
    spike_times = st[:,0].astype('int64') + imin  # shift by minimum sample index
    spike_templates = st[:,1].astype('int32')
    spike_clusters = clu
    xs, ys = compute_spike_positions(st, tF, ops)
    spike_positions = np.vstack([xs, ys]).T
    amplitudes = ((tF**2).sum(axis=(-2,-1))**0.5).cpu().numpy()
    # remove duplicate (artifact) spikes
    spike_times, spike_clusters, kept_spikes = remove_duplicates(
        spike_times, spike_clusters, dt=ops['settings']['duplicate_spike_bins']
    )
    amp = amplitudes[kept_spikes]
    spike_templates = spike_templates[kept_spikes]
    spike_positions = spike_positions[kept_spikes]
    np.save((results_dir / 'spike_times.npy'), spike_times)
    np.save((results_dir / 'spike_templates.npy'), spike_clusters)
    np.save((results_dir / 'spike_clusters.npy'), spike_clusters)
    np.save((results_dir / 'spike_positions.npy'), spike_positions)
    np.save((results_dir / 'spike_detection_templates.npy'), spike_templates)
    np.save((results_dir / 'amplitudes.npy'), amp)
    # Save spike mask so that it can be applied to other variables if needed
    # when loading results.
    np.save((results_dir / 'kept_spikes.npy'), kept_spikes)

    # template properties
    similar_templates = CCG.similarity(Wall, ops['wPCA'].contiguous(), nt=ops['nt'])
    template_amplitudes = ((Wall**2).sum(axis=(-2,-1))**0.5).cpu().numpy()
    templates = (Wall.unsqueeze(-1).cpu() * ops['wPCA'].cpu()).sum(axis=-2).numpy()
    templates = templates.transpose(0,2,1)
    templates_ind = np.tile(np.arange(Wall.shape[1])[np.newaxis, :], (templates.shape[0],1))
    np.save((results_dir / 'similar_templates.npy'), similar_templates)
    np.save((results_dir / 'templates.npy'), templates)
    np.save((results_dir / 'templates_ind.npy'), templates_ind)
    
    # pc features
    if save_extra_vars:
        # Save tF first since it gets updated in-place
        np.save(results_dir / 'tF.npy', tF.cpu().numpy())
    # This will momentarily copy tF which is pretty large, but it's on CPU
    # so the extra memory hopefully won't be an issue.
    tF = tF[kept_spikes]
    pc_features, pc_feature_ind = make_pc_features(
        ops, spike_templates, spike_clusters, tF
        )
    np.save(results_dir / 'pc_features.npy', pc_features)
    np.save(results_dir / 'pc_feature_ind.npy', pc_feature_ind)

    # contamination ratio
    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    is_ref, est_contam_rate = CCG.refract(spike_clusters, spike_times / ops['fs'],
                                          acg_threshold=acg_threshold,
                                          ccg_threshold=ccg_threshold)

    # write properties to *.tsv
    stypes = ['ContamPct', 'Amplitude', 'KSLabel']
    ks_labels = [['mua', 'good'][int(r)] for r in is_ref]
    props = [est_contam_rate*100, template_amplitudes, ks_labels]
    for stype, prop in zip(stypes, props):
        with open((results_dir / f'cluster_{stype}.tsv'), 'w') as f:
            f.write(f'cluster_id\t{stype}\n')
            for i,p in enumerate(prop):
                if stype != 'KSLabel':
                    f.write(f'{i}\t{p:.1f}\n')
                else:
                    f.write(f'{i}\t{p}\n')
        if stype == 'KSLabel':
            shutil.copyfile((results_dir / f'cluster_{stype}.tsv'), 
                            (results_dir / f'cluster_group.tsv'))

    # params.py
    dtype = "'int16'" if data_dtype is None else f"'{data_dtype}'"
    params = {'dat_path': f"'{Path(ops['settings']['filename']).resolve().as_posix()}'",
            'n_channels_dat': ops['settings']['n_chan_bin'],
            'dtype': dtype,
            'offset': 0,
            'sample_rate': ops['settings']['fs'],
            'hp_filtered': False }
    with open((results_dir / 'params.py'), 'w') as f: 
        for key in params.keys():
            f.write(f'{key} = {params[key]}\n')

    if save_extra_vars:
        # Also save Wall, for easier debugging/analysis
        np.save(results_dir / 'Wall.npy', Wall.cpu().numpy())
        # And full st, clu, amp arrays with no spikes removed
        np.save(results_dir / 'full_st.npy', st)
        np.save(results_dir / 'full_clu.npy', clu)
        np.save(results_dir / 'full_amp.npy', amplitudes)

    # Remove cached .phy results if present from running Phy on a previous
    # version of results in the same directory.
    phy_cache_path = Path(results_dir / '.phy')
    if phy_cache_path.is_dir():
        shutil.rmtree(phy_cache_path)

    return results_dir, similar_templates, is_ref, est_contam_rate, kept_spikes


def save_ops(ops, results_dir=None):
    """Save intermediate `ops` dictionary to `results_dir/ops.npy`."""

    if results_dir is None:
        results_dir = Path(ops['data_dir']) / 'kilosort4'
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    ops = ops.copy()
    # Convert paths to strings before saving, otherwise ops can only be loaded
    # on the system that originally ran the code (causes problems for tests).
    ops['settings']['results_dir'] = str(results_dir)
    # TODO: why do these get saved twice?
    ops['filename'] = str(ops['filename'])
    ops['data_dir'] = str(ops['data_dir'])
    ops['settings']['filename'] = str(ops['settings']['filename'])
    ops['settings']['data_dir'] = str(ops['settings']['data_dir'])

    # Convert pytorch tensors to numpy arrays before saving, otherwise loading
    # ops on a different system may not work (if saved from GPU, but loaded
    # on a system with only CPU).
    ops['is_tensor'] = []
    for k, v in ops.items():
        if isinstance(v, torch.Tensor):
            ops[k] = v.cpu().numpy()
            ops['is_tensor'].append(k)
    ops['preprocessing'] = {k: v.cpu().numpy()
                            for k, v in ops['preprocessing'].items()}

    np.save(results_dir / 'ops.npy', np.array(ops))


def load_ops(ops_path, device=None):
    """Load a saved `ops` dictionary and convert some arrays to tensors."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    ops = np.load(ops_path, allow_pickle=True).item()
    for k, v in ops.items():
        if k in ops['is_tensor']:
            ops[k] = torch.from_numpy(v).to(device)
    # TODO: Why do we have one copy of this saved as numpy, one as tensor,
    #       at different levels?
    ops['preprocessing'] = {k: torch.from_numpy(v).to(device)
                            for k,v in ops['preprocessing'].items()}

    return ops


class BinaryRWFile:

    supported_dtypes = ['int16', 'uint16', 'int32', 'float32']

    def __init__(self, filename: str, n_chan_bin: int, fs: int = 30000, 
                 NT: int = 60000, nt: int = 61, nt0min: int = 20,
                 device: torch.device = None, write: bool = False,
                 dtype: str = None, tmin: float = 0.0, tmax: float = np.inf,
                 file_object=None):
        """
        Creates/Opens a BinaryFile for reading and/or writing data that acts like numpy array

        * always assume int16 files *

        adapted from https://github.com/MouseLand/suite2p/blob/main/suite2p/io/binary.py
        
        Parameters
        ----------
        filename : str or Path
            The filename of the file to read from or write to
        n_chan_bin : int
            number of channels
        file_object : array-like file object; optional.
            Must have 'shape' and 'dtype' attributes and support array-like
            indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
            array or memmap.

        """
        self.fs = fs
        self.n_chan_bin = n_chan_bin
        self.filename = filename
        self.NT = NT 
        self.nt = nt 
        self.nt0min = nt0min
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device
        self.uint_set_warning = True
        self.writable = write

        if file_object is not None:
            dtype = file_object.dtype
        if dtype is None:
            dtype = 'int16'
        self.dtype = dtype

        if str(self.dtype) not in self.supported_dtypes:
            message = f"""
                {self.dtype} is not supported and may result in unexpected
                behavior or errors. Supported types are:\n
                {self.supported_dtypes}
                """
            warnings.warn(message, RuntimeWarning)

        # Must come after dtype since dtype is necessary for nbytesread
        if file_object is None:
            total_samples = get_total_samples(filename, n_chan_bin, dtype)
        else:
            n, c = file_object.shape
            assert c == n_chan_bin
            total_samples = n

        self.imin = max(int(tmin*fs), 0)
        self.imax = total_samples if tmax==np.inf else min(int(tmax*fs), total_samples)

        self.n_batches = int(np.ceil(self.n_samples / self.NT))

        mode = 'w+' if write else 'r'
        # Must use total samples for file shape, otherwise the end of the data
        # gets cut off if tmin,tmax are set.
        if file_object is not None:
            # For an already-loaded array-like file object,
            # such as a NumPy memmap
            self.file = file_object
        else:
            self.file = np.memmap(self.filename, mode=mode, dtype=self.dtype,
                                  shape=(total_samples, self.n_chan_bin))

    @property
    def n_samples(self) -> int:
        """total number of samples in the file."""
        return self.imax - self.imin

    @property
    def shape(self) -> Tuple[int, int]:
        """
        The dimensions of the data in the file
        Returns
        -------
        n_samples: int
            number of samples
        n_chan_bin: int
            number of channels
        """
        return self.n_samples, self.n_chan_bin

    @property
    def size(self) -> int:
        """
        Returns the total number of data points

        Returns
        -------
        size: int
        """
        return np.prod(np.array(self.shape).astype(np.int64))

    def close(self) -> None:
        """
        Closes the file.
        """
        del(self.file)
        self.file = None
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __setitem__(self, *items):
        if not self.writable:
            raise ValueError('Binary file was loaded as read-only.')

        idx, data = items
        # Shift indices by minimum sample index
        sample_indices = self._get_shifted_indices(idx)
        # Shift data to pos-only
        if self.dtype == 'uint16':
            data = data + 2**15
            if self.uint_set_warning:
                # Inform user of shift to hopefully avoid confusion, but only
                # do this once per bfile.
                print("NOTE: When setting new values for uint16 data, 2**15 will "
                      "be added to the given values before writing to file.")
                self.uint_set_warning = False
        # Convert back from float to file dtype
        data = data.astype(self.dtype)
        self.file[sample_indices] = data
        
    def __getitem__(self, *items):
        if self.file is None:
            raise ValueError('Binary file has been closed, data not accessible.')

        idx, *crop = items
        # Shift indices by minimum sample index.
        sample_indices = self._get_shifted_indices(idx)
        samples = self.file[sample_indices]
        # Shift data to +/- 2**15
        if self.dtype == 'uint16':
            samples = samples.astype('float32')
            samples = samples - 2**15

        return samples
    
    def _get_shifted_indices(self, idx):
        if not isinstance(idx, tuple): idx = tuple([idx])
        new_idx = []

        i = idx[0]
        if isinstance(i, slice):
            # Time dimension
            start = self.imin if i.start is None else i.start + self.imin
            stop = self.imax if i.stop is None else min(i.stop + self.imin, self.imax)
            new_idx.append(slice(start, stop, i.step))
        else:
            new_idx.append(i)

        if len(idx) == 2:
            # Channel dimension, should be no others after this.
            # No adjustments needed.
            new_idx.append(idx[1])

        return tuple(new_idx)

    def padded_batch_to_torch(self, ibatch, return_inds=False):
        """ read batches from file """
        if self.file is None:
            raise ValueError('Binary file has been closed, data not accessible.')

        if ibatch==0:
            bstart = self.imin
            bend = self.imin + self.NT + self.nt
        else:
            bstart = self.imin + (ibatch * self.NT) - self.nt
            bend = min(self.imax, bstart + self.NT + 2*self.nt)
        data = self.file[bstart : bend]
        data = data.T
        # Shift data to +/- 2**15
        if self.dtype == 'uint16':
            data = data.astype('float32')
            data = data - 2**15

        nsamp = data.shape[-1]
        X = torch.zeros((self.n_chan_bin, self.NT + 2*self.nt), device=self.device)

        with warnings.catch_warnings():
            # Don't need this, we know about the warning and it doesn't cause
            # any problems. Doing this the "correct" way is much slower.
            warnings.filterwarnings("ignore", message=_torch_warning)

            # fix the data at the edges for the first and last batch
            if ibatch == 0:
                X[:, self.nt : self.nt+nsamp] = torch.from_numpy(data).to(self.device).float()
                X[:, :self.nt] = X[:, self.nt : self.nt+1]
                bstart = self.imin - self.nt
            elif ibatch == self.n_batches-1:
                X[:, :nsamp] = torch.from_numpy(data).to(self.device).float()
                X[:, nsamp:] = X[:, nsamp-1:nsamp]
                bend += self.nt
            else:
                X[:] = torch.from_numpy(data).to(self.device).float()
    
        inds = [bstart, bend]
        if return_inds:
            return X, inds
        else:
            return X
        

def get_total_samples(filename, n_channels, dtype=np.int16):
    """Count samples in binary file given dtype and number of channels."""
    bytes_per_value = np.dtype(dtype).itemsize
    bytes_per_sample = np.int64(bytes_per_value * n_channels)
    total_bytes = os.path.getsize(filename)
    samples = (total_bytes / bytes_per_sample)

    if samples%1 != 0:
        raise ValueError(
            "Bytes in binary file did not divide evenly, "
            "incorrect n_chan_bin ('number of channels' in GUI)."
        )
    else:
        return int(samples)


class BinaryFileGroup:
    def __init__(self, file_objects):
        # NOTE: Assumes list order of files matches temporal order for
        #       concatenation.
        self.file_objects = file_objects
        self.dtype = file_objects[0].dtype
        self.n_chans = file_objects[0].shape[1]
        for f in file_objects[1:]:
            assert f.dtype == self.dtype, 'All files must have the same dtype'
            assert f.shape[1] == self.n_chans, \
                'All files must have the same number of channels'

        # Track indices that represent boundary between files. Each entry
        # is the starting index of the subsequent file.
        i = 0
        self.split_indices = []
        for f in file_objects:
            i += f.shape[0]
            self.split_indices.append(i)

    def __getitem__(self, *items):
        # Index into appropriate individual object based on index.
        # For indices that span multiple files, index all then concatenate result.
        idx, *crop = items
        if not isinstance(idx, tuple): idx = tuple([idx])
        channel_idx = idx[1] if len(idx) > 1 else slice(None)
        time_idx = idx[0]

        # To simplify for loop logic, convert integer index to slice and
        # convert NoneTypes to boundaries of data, and convert negative
        # indices to positive indices.
        if not isinstance(time_idx, slice):
            time_idx = slice(time_idx, time_idx+1)
        i = time_idx.start
        j = time_idx.stop

        if i is None: i = 0
        if i < 0: i = self.shape[0] + i
        if j is None: j = self.shape[0]
        if j < 0: j = self.shape[0] + j
        time_idx = slice(i, j)

        data = []
        shift = 0
        for k,f in zip(self.split_indices, self.file_objects):
            n_samples = f.shape[0]
            if time_idx.start < k:
                # At least part of the data is in this file
                t = slice(time_idx.start - shift, time_idx.stop - shift)
                data.append(f[t, channel_idx])
                if time_idx.stop <= k:
                    # This is the end of the data to be retrieved
                    break
            shift += n_samples

        if len(data) == 0:
            d = None
        elif len(data) == 1:
            d = data[0]
        else:
            d = np.concatenate(data, axis=0)
    
        return d

    @property
    def shape(self):
        return self.split_indices[-1], self.n_chans
    
    @staticmethod
    def from_filenames(filenames, n_channels, dtype=np.int16, mode='r'):
        files = []
        for name in filenames:
            n_samples = get_total_samples(name, n_channels, dtype)
            f = np.memmap(name, mode=mode, dtype=dtype,
                          shape=(n_samples, n_channels))
            files.append(f)
        
        return files


class BinaryFiltered(BinaryRWFile):
    def __init__(self, filename: str, n_chan_bin: int, fs: int = 30000, 
                 NT: int = 60000, nt: int = 61, nt0min: int = 20,
                 chan_map: np.ndarray = None, hp_filter: torch.Tensor = None,
                 whiten_mat: torch.Tensor = None, dshift: torch.Tensor = None,
                 device: torch.device = None, do_CAR: bool = True,
                 artifact_threshold: float = np.inf, invert_sign: bool = False,
                 dtype=None, tmin: float = 0.0, tmax: float = np.inf, file_object=None):

        super().__init__(filename, n_chan_bin, fs, NT, nt, nt0min, device,
                         dtype=dtype, tmin=tmin, tmax=tmax, file_object=file_object) 
        self.chan_map = chan_map
        self.whiten_mat = whiten_mat
        self.hp_filter = hp_filter
        self.dshift = dshift
        self.do_CAR = do_CAR
        self.invert_sign=invert_sign
        self.artifact_threshold = artifact_threshold

    def filter(self, X, ops=None, ibatch=None):
        # pick only the channels specified in the chanMap
        if self.chan_map is not None:
            X = X[self.chan_map]

        if self.invert_sign:
            X = X * -1

        X = X - X.mean(1).unsqueeze(1)
        if self.do_CAR:
            # remove the mean of each channel, and the median across channels
            X = X - torch.median(X, 0)[0]
    
        # high-pass filtering in the Fourier domain (much faster than filtfilt etc)
        if self.hp_filter is not None:
            fwav = fft_highpass(self.hp_filter, NT=X.shape[1])
            X = torch.real(ifft(fft(X) * torch.conj(fwav)))
            X = fftshift(X, dim = -1)

        if self.artifact_threshold < np.inf:
            if torch.any(torch.abs(X) >= self.artifact_threshold):
                # Assume the batch contains a recording artifact.
                # Skip subsequent preprocessing, zero-out the batch.
                return torch.zeros_like(X)

        # whitening, with optional drift correction
        if self.whiten_mat is not None:
            if self.dshift is not None and ops is not None and ibatch is not None:
                M = get_drift_matrix(ops, self.dshift[ibatch], device=self.device)
                #print(M.dtype, X.dtype, self.whiten_mat.dtype)
                X = (M @ self.whiten_mat) @ X
            else:
                X = self.whiten_mat @ X
        return X

    def __getitem__(self, *items):
        samples = super().__getitem__(*items)
        with warnings.catch_warnings():
            # Don't need this, we know about the warning and it doesn't cause
            # any problems. Doing this the "correct" way is much slower.
            warnings.filterwarnings("ignore", message=_torch_warning)
            X = torch.from_numpy(samples.T).to(self.device).float()
        return self.filter(X)
        
    def padded_batch_to_torch(self, ibatch, ops=None, return_inds=False):
        if return_inds:
            X, inds = super().padded_batch_to_torch(ibatch, return_inds=return_inds)
            return self.filter(X, ops, ibatch), inds
        else:
            X = super().padded_batch_to_torch(ibatch)
            return self.filter(X, ops, ibatch)


def spikeinterface_to_binary(recording, filepath, data_name='data.bin',
                             dtype=np.int16, chunksize=300000, export_probe=True,
                             probe_name='probe.prb', max_workers=None):
    """Save data from a SpikeInterface RecordingExtractor to a binary file.

    This function is provided to assist with converting data from other file
    formats to the raw binary format supported by Kilosort4. We do not explicitly
    support any other file format, nor is SpikeInterface a required package for
    using Kilosort4, so future updates to SpikeInterface may not be reflected here.
    If you run into errors and would like help loading your data into Kilosort4,
    please post an issue on the Kilosort4 github.

    Parameters
    ----------
    recording : RecordingExtractor
        A SpikeInterface object containing the recording to be copied.
    filepath : str or Path-like.
        Path to the directory where the binary file (and possibly prb file)
        should be saved.
    data_name : str; default='data.bin'
        Name for the new binary file.
    dtype : type; default=np.int16
        Data type for the new binary file.
    chunksize : int; default=60000.
        Number of samples to copy on each loop. A higher number may speed up
        copying, but will use more memory.
    export_probe : bool; default=True.
        If True, attempt to extract probe information from the recording and
        write it to a .prb file.
    probe_name : str; default='probe.prb'
        Name for the new probe file.
    max_workers : int; optional.
        Maximum number of threads used to execute file i/o.
        Default: min(32, (os.process_cpu_count() or 1) + 4)
        (https://github.com/python/cpython/blob/main/Lib/concurrent/futures/thread.py)
        
    Notes
    -----
    SpikeInterface has its own `recording.save()` method for this purpose
    that supports parallelization. This simpler utility is provided for
    better control over filepath structure, minimal output, and fewer
    dependencies on file format details. However, for very large files, you
    may want to investigate `recording.save` to speed up data copying.
    
    """

    filepath = Path(filepath)
    filepath.mkdir(exist_ok=True, parents=True)
    binary_filename = filepath / f'{data_name}'
    probe_filename = filepath / f'{probe_name}'

    print('='*40)
    print('Loading recording with SpikeInterface...')

    # Using actual data shape is less fragile than relying on .get_num_channels()
    N = recording.get_total_samples()
    print(f'number of samples: {N}')
    c = recording.get_traces(start_frame=0, end_frame=1, segment_index=0).shape[1]
    print(f'number of channels: {c}')
    s = recording.get_num_segments()
    print(f'numbef of segments: {s}')
    fs = recording.get_sampling_frequency()
    print(f'sampling rate: {fs}')
    dtype = recording.get_dtype()
    print(f'dtype: {dtype}')

    # Determine start/end indices for each segment
    indices = []
    for k in range(s):
        n = recording.get_num_samples(segment_index=k)
        i = 0 + k*chunksize
        while i < n:
            j = i + chunksize if (i + chunksize) < n else n
            indices.append((i, j, k))
            i += chunksize

    # Copy each chunk of data to memmory mapped binary file,
    # use multithreading to speed it up.
    def copy_chunk(memmap, i, j, k):
        t = recording.get_traces(start_frame=i, end_frame=j, segment_index=k)
        memmap[i:j,:] = t
        memmap.flush()
        del(t)

    y = np.memmap(binary_filename, dtype=dtype, mode='w+', shape=(N,c))
    total_chunks = len(indices)
    print('='*40)
    print(f'Converting {total_chunks} data chunks '
          f'with a chunksize of {chunksize} samples...')
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(copy_chunk, y, i, j, k) for i, j, k in indices]
        # TODO: every some-amount-of-time, check list of futures
        #       for completion
        while exe._work_queue.qsize() > 0:
            time.sleep(10)
            print(f'{total_chunks - exe._work_queue.qsize()} of {total_chunks}'
                   ' chunks converted...')
    print(f'Data conversion finished.')
    print('='*40)

    del(y)  # Close memmap after copying

    if export_probe:
        try:
            from probeinterface import write_prb
            try:
                pg = recording.get_probegroup()
                write_prb(probe_filename, pg)
            except ValueError:
                print(
                    'SpikeInterface recording contains no probe information,\n'
                    'could not write .prb file.'
                )
                probe_filename = None
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Could not import `write_prb` from probeinterface when exporting '
                'probe, please run `pip install spikeinterface`.'
            )
    else:
        probe_filename = None

    return binary_filename, N, c, s, fs, probe_filename


class RecordingExtractorAsArray:

    def __init__(self, recording_extractor):
        """An array-like wrapper for a RecordingExtractor.

        This class is provided to assist with loading data from other file 
        formats besides the raw binary format supported by Kilosort4. We do not
        explicitly support any other file format, nor is SpikeInterface a
        required package for using Kilosort4, so future updates to SpikeInterface
        may not be reflected here. If you run into errors and would like help
        loading your data into Kilosort4, please post an issue on the Kilosort4
        github.

        Parameters
        ----------
        recording_extractor : RecordingExtractor
            A SpikeInterface recording extractor. When the wrapper object is
            indexed, `recording_extractor.get_traces()` will be invoked to
            retrieve data from disk.

        Attributes
        ----------
        shape
        dtype

        Examples
        --------
        >>> from spikeinterface.extractors import read_nwb_recording
        >>> recording = read_nwb_recording('/home/my_file_path/arange.nwb')
        >>> as_array = RecordingExtractorAsArray(recording)
        >>> as_array[:5, 2]
        array([[0],
               [1],
               [2],
               [3],
               [4],
               [5]])

        """

        if recording_extractor.get_num_segments() > 1:
            try:
                import spikeinterface as si
                self.recording = si.concatenate_recordings([recording_extractor])
                print('SpikeInterface recording contains more than one segment, '
                      'segments will be concatenated as if contiguous.')
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'SpikeInterface could not be imported, but is needed for '
                    'loading multi-segment SpikeInterface recordings. Please '
                    'run `pip install spikeinterface`.'
                )

        print('='*40)
        print('Loading recording with SpikeInterface...')
        self.recording = recording_extractor
        self.N = self.recording.get_total_samples()
        print(f'number of samples: {self.N}')
        self.c = self.recording.get_traces(start_frame=0, end_frame=1, segment_index=0).shape[1]
        print(f'number of channels: {self.c}')
        self.s = self.recording.get_num_segments()
        print(f'numbef of segments: {self.s}')
        self.fs = self.recording.get_sampling_frequency()
        print(f'sampling rate: {self.fs}')
        self.dtype = self.recording.get_dtype()
        print(f'dtype: {self.dtype}')
        self.shape = (self.N, self.c)
        print('='*40)
    

    def __getitem__(self, *items):
        idx, *crop = items
        if not isinstance(idx, tuple): idx = tuple([idx])
        sample_idx = idx[0]
        channel_ids = None if len(idx) == 1 else idx[1]

        # Convert integer index to slice and convert NoneTypes to boundaries of
        # data, and convert negative indices to positive indices.
        if not isinstance(sample_idx, slice):
            sample_idx = slice(sample_idx, sample_idx+1)
        i = sample_idx.start
        j = sample_idx.stop

        if i is None: i = 0
        if i < 0: i = self.N + i
        if j is None: j = self.N
        if j < 0: j = self.N + j

        # Convert channel slice to list of indices starting from 0
        if isinstance(channel_ids, slice):
            c = channel_ids.start
            d = channel_ids.stop
            if c is None: c = 0
            if d is None: d = self.shape[1]
            channel_ids = list(range(c, d))
        elif channel_ids is not None:
            assert isinstance(channel_ids, int)
            channel_ids = [channel_ids]
        else:
            channel_ids = np.arange(self.shape[1])
        # Index into actual channel ids from recording, which do not have to 
        # be sequential or start from 0
        channel_ids = self.recording.channel_ids[channel_ids]

        samples = self.recording.get_traces(start_frame=i, end_frame=j,
                                            channel_ids=channel_ids)
        
        return samples

    def __setitem__(self):
        raise ValueError('RecordingExtractorAsBinary is read-only.')
