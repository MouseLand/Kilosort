from typing import Optional, Tuple, Sequence
from contextlib import contextmanager
import os
from glob import glob
from scipy.io import loadmat
import torch
import numpy as np
from torch.fft import fft, ifft, fftshift
from pathlib import Path
from kilosort.preprocessing import get_drift_matrix, fft_highpass

def find_binary(data_folder):
    """ find binary file in data_folder"""
    filename  = glob(os.path.join(data_folder, '*.bin'))[0]
    return filename

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
        probe['n_chan'] = len(probe['chanMap'])  # NOTE: should match the # of columns in the raw data

    for n in required_keys:
        assert n in probe.keys()

    return probe

class BinaryRWFile:
    def __init__(self, filename: str, n_chan_bin: int, fs: int = 30000, 
                batch_size: int = 60000, n_twav: int = 61,
                twav_min: int = 20, device: torch.device = torch.device('cpu'),
                write: bool = False):
        """
        Creates/Opens a BinaryFile for reading and/or writing data that acts like numpy array

        * always assume int16 files *

        adapted from https://github.com/MouseLand/suite2p/blob/main/suite2p/io/binary.py
        
        Parameters
        ----------
        filename: str
            The filename of the file to read from or write to
        n_chan_bin: int
            number of channels
        """
        self.fs = fs
        self.n_chan_bin = n_chan_bin
        self.filename = filename
        self.batch_size = batch_size 
        self.n_twav = n_twav 
        self.twav_min = twav_min
        self.device = device
        if write:
            self.file = open(filename, mode='w+b')
        else:
            self.file = open(filename, mode='r+b')
        self._index = 0
        self._can_read = True
        self.n_batches = self.n_samples // self.batch_size + 1

    @staticmethod
    def convert_numpy_file_to_binary(from_filename: str, to_filename: str) -> None:
        """
        Works with npz files, pickled npy files, etc.
        Parameters
        ----------
        from_filename: str
            The npy file to convert
        to_filename: str
            The binary file that will be created
        """
        np.load(from_filename).tofile(to_filename)

    @property
    def nbytesread(self):
        """number of bytes per sample (FIXED for given file)"""
        return np.int64(2 * self.n_chan_bin)

    @property
    def nbytes(self):
        """total number of bytes in the file."""
        with temporary_pointer(self.file) as f:
            f.seek(0, 2)
            return f.tell()

    @property
    def n_samples(self) -> int:
        """total number of samples in the file."""
        return int(self.nbytes // self.nbytesread)

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
        self.file.close()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __setitem__(self, *items):
        sample_indices, data = items
        self.ix_write(data=data, indices=self.from_slice(sample_indices))
        
    def __getitem__(self, *items):
        sample_indices, *crop = items
        if isinstance(sample_indices, int):
            samples = self.ix(indices=[sample_indices], is_slice=False)
        elif isinstance(sample_indices, slice):
            samples = self.ix(indices=self.from_slice(sample_indices), is_slice=True)
        else:
            samples = self.ix(indices=sample_indices, is_slice=False)
        return samples[(slice(None),) + crop] if crop else samples

    def ix_write(self, data, indices: Sequence[int]):
        """
        Writes the samples at index values "indices".
        Parameters
        ----------
        indices: int array
            The sample indices to get, must be a slice
        """
        i0 = indices[0]
        batch_size = len(indices)
        if self._index != i0:
            self.file.seek(self.nbytesread * (i0 - self._index), 1)
        self._index = i0 + batch_size
        self.write(data)  

    def ix(self, indices: Sequence[int], is_slice=False):
        """
        Returns the samples at index values "indices".
        Parameters
        ----------
        indices: int array
            The sample indices to get
        is_slice: bool, default False
            if indices are slice, read slice with "read" function and return
        Returns
        -------
        samples: len(indices) x n_chan_bin
            The requested samples
        """
        if not is_slice:
            samples = np.empty((len(indices), self.n_chan_bin), np.int16)
            # load and bin data
            with temporary_pointer(self.file) as f:
                for sample, ixx in zip(samples, indices):
                    if ixx!=self._index:
                        f.seek(self.nbytesread * ixx)
                    buff = f.read(self.nbytesread)
                    data = np.frombuffer(buff, dtype=np.int16, offset=0)
                    sample[:] = data
                    #self._index = ixx+1
        else:
            i0 = indices[0]
            batch_size = len(indices)
            if self._index != i0:
                self.file.seek(self.nbytesread * i0)
            _, samples = self.read(batch_size=batch_size)
            self._index = i0 + batch_size
        
        return samples

    def read(self, batch_size=1) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the next sample(s) in the file and its associated indices.
        Parameters
        ----------
        batch_size: int
            The number of samples to read at once.
        samples: batch_size x n_chan_bin
            The sample data
        """
        if not self._can_read:
            raise IOError("BinaryFile needs to write before it can read again.")
        nbytes = self.nbytesread * batch_size
        buff = self.file.read(nbytes)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).reshape(-1, self.n_chan_bin)
        if data.size == 0:
            return None
        indices = np.arange(self._index, self._index + data.shape[0])
        self._index += data.shape[0]
        return indices, data

    def write(self, data: np.ndarray) -> None:
        """
        Writes sample(s) to the file.
        Parameters
        ----------
        data: 2D or 3D array
            The sample(s) to write.  Should be the same width and height as the other samples in the file.
        """
        self.file.write(bytearray(np.minimum(data, 2 ** 15 - 2).astype('int16')))


    def from_slice(self, s: slice) -> Optional[np.ndarray]:
        """Creates an np.arange() array from a Python slice object.  Helps provide numpy-like slicing interfaces."""
        s_start = 0 if s.start is None else s.start
        s_step = 1 if s.step is None else s.step 
        s_stop = self.n_samples if s.stop is None else s.stop
        s_start = self.n_samples + s_start if s_start < 0 else s_start
        s_stop = self.n_samples + s_stop if s_stop < 0 else s_stop
        s_stop = min(self.n_samples, s_stop)
        return np.arange(s_start, s_stop, s_step) if any([s_start, s_stop, s_step]) else None

    def padded_batch_to_torch(self, ibatch):
        """ read batches from file """
        if ibatch==0:
            bstart = 0
            bend = self.batch_size + self.n_twav
        else:
            bstart = ibatch * self.batch_size - self.n_twav
            bend = min(self.n_samples, bstart + self.batch_size + 2*self.n_twav)
        data = self.ix(np.arange(bstart, bend), is_slice=True)        
        data = data.T
        nsamp = data.shape[-1]
        X = torch.zeros((self.n_chan_bin, self.batch_size + 2*self.n_twav), device = self.device)
        # fix the data at the edges for the first and last batch
        if ibatch==0:
            X[:, self.n_twav : self.n_twav+nsamp] = torch.from_numpy(data).to(self.device).float()
            X[:, :self.n_twav] = X[:, self.n_twav : self.n_twav+1]
        elif ibatch==self.n_batches-1:
            X[:, :nsamp] = torch.from_numpy(data).to(self.device).float()
            X[:, nsamp:] = X[:, nsamp-1:nsamp]
        else:
            X[:] = torch.from_numpy(data).to(self.device).float()
        return X 

class BinaryFiltered(BinaryRWFile):
    def __init__(self, filename: str, n_chan_bin: int, fs: int = 30000, 
                batch_size: int = 60000, n_twav: int = 61,
                twav_min: int = 20, channel_map: np.ndarray = None, 
                hp_filter: torch.Tensor = None, whiten_mat: torch.Tensor = None, 
                dshift: torch.Tensor = None, device: torch.device = torch.device('cuda')):
        super().__init__(filename, n_chan_bin, fs, batch_size, n_twav, twav_min, device) 
        self.channel_map = channel_map
        self.whiten_mat = whiten_mat
        self.hp_filter = hp_filter
        self.dshift = dshift

    def filter(self, X, ops=None, ibatch=None):
        # pick only the channels specified in the chanMap
        if self.channel_map is not None:
            X = X[self.channel_map]

        # remove the mean of each channel, and the median across channels
        X = X - X.mean(1).unsqueeze(1)
        X = X - torch.median(X, 0)[0]
    
        # high-pass filtering in the Fourier domain (much faster than filtfilt etc)
        if self.hp_filter is not None:
            fwav = fft_highpass(self.hp_filter, NT=X.shape[1])
            X = torch.real(ifft(fft(X) * torch.conj(fwav)))
            X = fftshift(X, dim = -1)

        # whitening, with optional drift correction
        if self.whiten_mat is not None:
            if self.dshift is not None and ops is not None and ibatch is not None:
                M = get_drift_matrix(ops, self.dshift[ibatch])
                #print(M.dtype, X.dtype, self.whiten_mat.dtype)
                X = (M @ self.whiten_mat) @ X
            else:
                X = self.whiten_mat @ X
        return X

    def __getitem__(self, *items):
        sample_indices, *crop = items
        if isinstance(sample_indices, int):
            samples = self.ix(indices=[sample_indices], is_slice=False)
        elif isinstance(sample_indices, slice):
            samples = self.ix(indices=self.from_slice(sample_indices), is_slice=True)
        else:
            samples = self.ix(indices=sample_indices, is_slice=False)
        samples = samples[(slice(None),) + crop] if crop else samples
        X = torch.from_numpy(samples.T).to(self.device).float()
        return self.filter(X)
        
    def padded_batch_to_torch(self, ibatch, ops=None):
        X = super().padded_batch_to_torch(ibatch)        
        return self.filter(X, ops, ibatch)


@contextmanager
def temporary_pointer(file):
    """context manager that resets file pointer location to its original place upon exit."""
    orig_pointer = file.tell()
    yield file
    file.seek(orig_pointer)