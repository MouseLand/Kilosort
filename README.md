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


## Technical notes about the port

The following differences between MATLAB and Python required special care during the port:

* Discrepancy between 0-based and 1-based indexing.
* MATLAB uses Fortran ordering for arrays, whereas NumPy uses C ordering by default. The Python code therefore uses Fortran ordering exclusively so that the custom CUDA kernels can be used with no modification.
* In MATLAB, arrays can be extended transparently with indexing, whereas NumPy/CuPy requires explicit concatenation.
* The MATLAB code used mex C files to launch CUDA kernels, whereas the Python code uses CuPy directly.
* A few workarounds around limitations of CuPy compared to MATLAB: no `cp.median()`, no GPU version of the `lfilter()` LTI filter in CuPy (a custom CUDA kernel had to be written), etc.
