# Kilosort4

[![Documentation Status](https://readthedocs.org/projects/kilosort/badge/?version=latest)](https://kilosort.readthedocs.io/en/latest/?badge=latest)
![tests](https://github.com/mouseland/kilosort/actions/workflows/test_and_deploy.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/kilosort.svg)](https://badge.fury.io/py/kilosort)
[![Downloads](https://pepy.tech/badge/kilosort)](https://pepy.tech/project/kilosort)
[![Downloads](https://pepy.tech/badge/kilosort/month)](https://pepy.tech/project/kilosort)
[![Python version](https://img.shields.io/pypi/pyversions/kilosort)](https://pypistats.org/packages/kilosort)
[![Licence: GPL v3](https://img.shields.io/github/license/MouseLand/kilosort)](https://github.com/MouseLand/kilosort/blob/master/LICENSE)
[![Contributors](https://img.shields.io/github/contributors-anon/MouseLand/kilosort)](https://github.com/MouseLand/kilosort/graphs/contributors)
[![repo size](https://img.shields.io/github/repo-size/MouseLand/kilosort)](https://github.com/MouseLand/kilosort/)
[![GitHub stars](https://img.shields.io/github/stars/MouseLand/kilosort?style=social)](https://github.com/MouseLand/kilosort/)
[![GitHub forks](https://img.shields.io/github/forks/MouseLand/kilosort?style=social)](https://github.com/MouseLand/kilosort/)


You can run Kilosort4 without installing it locally using google colab. An example colab notebook is available [here](https://colab.research.google.com/drive/1gFZa8TEBDXmg_CB5RwuT_52Apl3hP0Ie?usp=sharing). It will download some test data, run kilosort4 on it, and show some plots.

Example notebooks are provided in the `docs/source/tutorials` folder and will be later published to readthedocs. The notebooks include: 

  1. `basic_example`:  sets up run on example data and shows how to modify parameters  
  2. `load_data`:  example data format conversion through SpikeInterface  
  3. `make_probe`:  making a custom probe configuration. 

# Installation

### System requirements

Linux and Windows 64-bit are supported for running the code. At least 8GB of GPU RAM is required to run the software. The software has been tested on Windows 10 and Ubuntu 20.04. 

### Instructions

If you have an older `kilosort` environment you can remove it with `conda env remove -n kilosort` before creating a new one.

1. Install an [Anaconda](https://www.anaconda.com/products/distribution) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt which has `conda` for **python 3** in the path
3. Create a new environment with `conda create --name kilosort python=3.8`. We recommend python 3.8 or python 3.9 but 3.10 should work as well.
4. To activate this new environment, run `conda activate kilosort`
5. Change into the directory containing the code, it should have a `setup.py` file in it.
6. To install kilosort and the GUI, run `python -m pip install .[gui]`. If you're on a zsh server, you may need to use ' ' around the kilosort[gui] call: `python -m pip install '.[gui]'.
7. Instead of 6, you can install the minimal version of kilosort with `python -m pip install .`.  
8. Next remove the CPU version of pytorch `pip uninstall torch`
9. Then install the GPU version of pytorch `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia`

Note you will always have to run `conda activate kilosort` before you run kilosort. If you want to run jupyter notebooks in this environment, then also `conda install jupyter` or `pip install notebook`, and `python -m pip install matplotlib`.

### Debugging pytorch installation 

If step 9 does not work, you need to make sure the NVIDIA driver for your GPU is installed (available [here](https://www.nvidia.com/Download/index.aspx?lang=en-us)). You may also need to install the CUDA libraries for it, we recommend [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive).

If pytorch installation still fails, follow the instructions [here](https://pytorch.org/get-started/locally/) to determine what version of pytorch to install. The Anaconda install is strongly recommended, and then choose the CUDA version that is supported by your GPU (newer GPUs may need newer CUDA versions > 10.2). For instance this command will install the 11.8 version on Linux and Windows (note the `torchvision` and `torchaudio` commands are removed because kilosort doesn't require them):

``
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
``

This [video](https://www.youtube.com/watch?v=gsixIQYvj3U) has step-by-step installation instructions for NVIDIA drivers and pytorch in Windows (ignore the environment creation step with the .yml file, we have an environment already, to activate it use `conda activate kilosort`).


# Running kilosort 

1. Open the GUI with `python -m kilosort`
2. Select the path to the binary file and optionally the results directory. We recommend putting the binary file on an SSD for faster processing. 
3. Select the probe configuration (mat files recommended, they actually exclude off channels unlike prb files)
4. Hit `LOAD`. The data should now be visible.
5. Hit `Run`. This will run the pipeline and output the results in a format compatible with Phy, the most popular spike sorting curating software.

There is a warning that will always pop up when running Kilosort and/or using the BinaryFile class, but it's nothing to worry about:
```
UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at C:\cb\pytorch_1000000000000\work\torch\csrc\utils\tensor_numpy.cpp:205.)
```

## Integration with Phy GUI

[Phy](https://github.com/kwikteam/phy) provides a manual clustering interface for refining the results of the algorithm. Kilosort4 automatically sets the "good" units in Phy based on a <10% estimated contamination rate with spikes from other neurons (computed from the refractory period violations relative to expected).

Check out the [Phy](https://github.com/kwikteam/phy) repo for more in-depth install instructions, but in most cases the following should work: activate the kilosort environment (`conda activate kilosort`), and then
~~~
pip install phy --pre --upgrade
~~~

Note there is a deprecation by numpy that will break phy, so please `pip install numpy==1.23`.

Next change to the results directory from kilosort4 (by default a folder named `kilosort4` in the binary directory) and run:
~~~
phy template-gui params.py
~~~

Now phy should run correctly in the `kilosort` environment you made (if not make a new environment for phy).

### Developer instructions

Need to install pytest
~~~
pip install pytest
~~~

Then run all tests with:
~~~
pytest tests/ --runslow

