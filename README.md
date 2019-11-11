# [WIP] Python port of KiloSort2

This is a work-in-progress litteral Python port of the original MATLAB version of Kilosort 2, written by Marius Pachitariu.
The code is still being debugged and is not ready for use.


## Hardware requirements

The code makes extensive use of the GPU via the CUDA framework. A high-end GPU with at least 8GB of memory is required.
A good CPU and a large amount of RAM (minimum 32GB or 64GB) is also required.


## Dependencies

* Python 3.7+
* NumPy
* SciPy
* CuPy
* matplotlib
* tqdm
* click
* pytest


## Usage example

The programming interface is subject to change. The following code example should be saved in a directory, along with the following files:

* imec_385_100s.bin
* chanMap.npy (be careful with 0- and 1- indexing discrepancy between MATLAB and Python: don't forget to subtract 1 if this file was used in Kilosort)
* xc.npy
* yc.npy
* kcoords.npy

```python
from pathlib import Path
import numpy as np

from pykilosort import add_default_handler, run, Bunch, read_data

add_default_handler(level='DEBUG')

dat_path = Path('imec_385_100s.bin')
dir_path = dat_path.parent

probe = Bunch()
probe.NchanTOT = 385

# WARNING: indexing mismatch with MATLAB hence the -1
probe.chanMap = np.load(dir_path / 'chanMap.npy').squeeze().astype(np.int64) - 1

probe.xc = np.load(dir_path / 'xc.npy').squeeze()
probe.yc = np.load(dir_path / 'yc.npy').squeeze()
probe.kcoords = np.load(dir_path / 'kcoords.npy').squeeze()

# The raw data file is loaded in Fortran order, its shape is therefore (n_channels, n_samples).
raw_data = read_data(dat_path, shape=(probe.NchanTOT, -1), dtype=np.int16)

run(dir_path, raw_data, probe, dat_path=dat_path)

```


## Disk cache

The MATLAB version used a big `rez` structured object containing the input data, the parameters, intermediate and final results.

The Python version makes the distinction beteween:

- `raw_data`: a NumPy-like object of shape `(n_samples, n_channels_total)`
- `probe`: a Bunch instance (dictionary) with the channel coordinates, the indices of the "good channels"
- `params`: a Bunch instance (dictionary) with optional user-defined parameters. It can be empty. Any missing parameter is transparently replaced by the default as found in `default_params.py` file in the repository.
- `intermediate`: a Bunch instance (dictionary) with intermediate arrays.

These objects are accessible via the *context* (`ctx`) which replaces the MATLAB `rez` object: `ctx.raw_data`, etc.

This context also stores a special object called `ctx.intermediate` which stores intermediate arrays. This object derives from `Bunch` and implements special methods to save and load arrays in a temporary folder. By default, an intermediate result called `ctx.intermediate.myarray` is stored in `./.kilosort/context/myarray.npy`.

The main `run()` function checks the existence of some of these intermediate arrays to skip some steps that might have run already, for a given dataset.


## Technical notes about the port

The following differences between MATLAB and Python required special care during the port:

* Discrepancy between 0-based and 1-based indexing.
* MATLAB uses Fortran ordering for arrays, whereas NumPy uses C ordering by default. The Python code therefore uses Fortran ordering exclusively so that the custom CUDA kernels can be used with no modification.
* In MATLAB, arrays can be extended transparently with indexing, whereas NumPy/CuPy requires explicit concatenation.
* The MATLAB code used mex C files to launch CUDA kernels, whereas the Python code uses CuPy directly.
* A few workarounds around limitations of CuPy compared to MATLAB: no `cp.median()`, no GPU version of the `lfilter()` LTI filter in CuPy (a custom CUDA kernel had to be written), etc.
