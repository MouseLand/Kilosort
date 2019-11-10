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

```python
from pykilosort import run

# TODO
run(dat_path=dat_path, raw_data=raw_data, probe=probe, params=params)
```


## Disk cache

The MATLAB version used a big `rez` structured object containing the input data, the parameters, intermediate and final results.

The Python version makes the distinction beteween:

- `raw_data`: a NumPy-like object of shape `(n_samples, n_channels_total)`
- `probe`: a Bunch instance (dictionary) with the channel coordinates, the indices of the "good channels"
- `params`: a Bunch instance (dictionary) with optional user-defined parameters. It can be empty. Any missing parameter is transparently replaced by the default as found in `default_params.py` file in the repository.

These objects are accessible via the *context* (`ctx`) which replaces the MATLAB `rez` object: `ctx.raw_data`, etc. This context also stores a special object called `ctx.intermediate` which stores intermediate arrays. This object derives from `Bunch` and implements special methods to save and load arrays in a temporary folder. By default, an intermediate array called `ctx.intermediate.myarray` is stored in `./.kilosort/context/myarray.npy`.

The main `run()` function checks the existence of some of these intermediate arrays to skip some steps that might have run already, for a given dataset.


## Technical notes about the port

The following differences between MATLAB and Python required special care during the port:

* Discrepancy between 0-based and 1-based indexing.
* MATLAB uses Fortran ordering for arrays, whereas NumPy uses C ordering by default. The Python code therefore uses Fortran ordering exclusively so that the custom CUDA kernels can be used with no modification.
* In MATLAB, arrays can be extended transparently with indexing, whereas NumPy/CuPy requires explicit concatenation.
* The MATLAB code used mex C files to launch CUDA kernels, whereas the Python code uses CuPy directly.
* A few workarounds around limitations of CuPy compared to MATLAB: no `cp.median()`, no GPU version of the `lfilter()` LTI filter in CuPy (a custom CUDA kernel had to be written), etc.
