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


You can run Kilosort4 without installing it locally using google colab. An example colab notebook is available here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mouseland/kilosort/blob/main/docs/tutorials/kilosort4.ipynb). It will download some test data, run kilosort4 on it, and show some plots. Talk describing Kilosort4 is [here](https://www.youtube.com/watch?v=LTSmoACr918). 

Example notebooks are provided in the `docs/source/tutorials` folder and in the [docs](https://kilosort.readthedocs.io/en/latest/tutorials/tutorials.html). The notebooks include: 

  1. `basic_example`:  sets up run on example data and shows how to modify parameters  
  2. `load_data`:  example data format conversion through SpikeInterface  
  3. `make_probe`:  making a custom probe configuration.

**If you use Kilosort1-4, please cite the [paper](https://www.nature.com/articles/s41592-024-02232-7):**     
Pachitariu, M., Sridhar, S., Pennington, J., & Stringer, C. (2024). Spike sorting with Kilosort4. _Nature Methods_ , 21, pages 914–921

**Warning**: There were two bugs in Kilosort 2, 2.5 and 3 (but not 4) which caused fewer spikes to be detected in ~7ms periods at batch boundaries (every 2.1866s, issue #594). The patch1 releases fix these bugs, please use the new default NT and ntbuff parameters. Also, we are no longer providing in-depth support for Kilosort 1-3. If you encounter difficulties using the older versions, we recommend trying Kilosort4 instead.


## System requirements

Linux and Windows 64-bit are supported for running the code. At least 8GB of GPU RAM is required to run the software, but 12GB or more is recommended. See [docs](https://kilosort.readthedocs.io/en/latest/hardware.html) for more recommendations.

The software has been fully tested on Windows 10, Windows 11, and Ubuntu 20.04. We also run a subset of tests on macOS 13 but can't guarantee full support.

## Instructions

These instructions are written for use with Anaconda distributions of python and the `conda` environment manager. We find this to be the most user-friendly approach. If you prefer to use a different environment manager and need help with installation, please create a new issue to ask for assistance.

1. Install an [Anaconda](https://www.anaconda.com/products/distribution) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt which has `conda` for **python 3** in the path.
3. Create a new environment with `conda create --name kilosort python=3.9`. Python 3.10 should work as well. If you have an older `kilosort` environment you can remove it with `conda env remove -n kilosort` before creating a new one.
4. To activate this new environment, run `conda activate kilosort`.
5. To install kilosort and the GUI, run `python -m pip install kilosort[gui]`. If you're on a zsh server, you may need to use `python -m pip install "kilosort[gui]" `.
6. Instead of step 5, you can install the minimal version of kilosort with `python -m pip install kilosort`.
7. To run Kilosort on GPU (strongly recommended), remove the CPU version of PyTorch with `pip uninstall torch` and proceed to 8. Otherwise, stop here.
8. Install the GPU version of PyTorch (for CUDA 11.8) with `pip3 install torch --index-url https://download.pytorch.org/whl/cu118`.

Note you will always have to run `conda activate kilosort` before you run kilosort. If you want to run jupyter notebooks in this environment, you will also need to `conda install jupyter` or `pip install notebook`.

## Running kilosort 

1. Open the GUI with `python -m kilosort`
2. Select the path to the binary file and optionally the results directory. We recommend putting the binary file on an SSD for faster processing. 
3. Select the probe configuration (mat files recommended, they actually exclude off channels unlike prb files)
4. Hit `LOAD`. The data should now be visible.
5. Hit `Run`. This will run the pipeline and output the results in a format compatible with Phy, the most popular spike sorting curating software.

Some things to be aware of when running Kilosort4:
* Re-using parameters from previous versions of Kilosort will probably not work well. Kilosort4 is a new algorithm, and the main parameters (like the detection thresholds) can affect the results in different ways. **Please start with the default parameters** and adjust from there based on what you see in Phy. For descriptions of Kilosort4's parameters, you can mouse-over their names in the GUI or look at `kilosort.parameters.py`. You can find [additional explanation for some parameters here](https://kilosort.readthedocs.io/en/latest/parameters.html).

* We do not provide support for SpikeInterface, and are not involved in their development (or vise-versa). If you encounter problems running Kilosort4 through SpikeInterface, please try running Kilosort4 directly instead. In particular, the KS4 GUI is a useful tool for checking that your probe and data are formatted correctly.


## Integration with Phy GUI

[Phy](https://github.com/cortex-lab/phy) provides a manual clustering interface for refining the results of the algorithm. Kilosort4 automatically sets the "good" units in Phy based on a <20% estimated contamination rate with spikes from other neurons (computed from the refractory period violations relative to expected). Check out the [Phy](https://github.com/cortex-lab/phy) repo for installation instructions. **We recommend installing Phy in its own environment to avoid package conflicts**.

After installation, activate your Phy environment and navigate to the results directory from Kilosort4 (by default, a folder named `kilosort4` in the same directory as the binary data file) and run:
~~~
phy template-gui params.py
~~~


## Debugging PyTorch installation 

If step 8 does not work, you need to make sure the NVIDIA driver for your GPU is installed ([check here](https://www.nvidia.com/Download/index.aspx?lang=en-us)). You may also need to install the CUDA libraries for it. We recommend [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) if it's compatible with your graphics card, since that is what we use for testing and development. If the installation still fails, follow the [instructions here](https://pytorch.org/get-started/locally/) to determine what version of PyTorch to install. Choose the CUDA version that is supported by your GPU (newer GPUs may need newer CUDA versions > 11.8), then run the suggested command in your Kilosort environment. For instance, this command will install the 12.4 version on Linux and Windows:

``
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
``

Some GPUs may work best with a CUDA toolkit version that Pytorch does not have a stable release for yet. For example, the RTX 5090 currently works best with CUDA version 12.8 which Pytorch does not yet fully support. You can install Pytorch builds sooner using their nightly releases, like the following, if you want to use the new version anyway:
~~~
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
~~~
So far, we haven't observed any issues using CUDA 12.8 with an RTX 5090 card using the nightly builds.

This [video](https://www.youtube.com/watch?v=gsixIQYvj3U) has step-by-step installation instructions for NVIDIA drivers and PyTorch in Windows (ignore the environment creation step with the .yml file, activate your existing environment with `conda activate kilosort`).


## Debugging qt.qpa.plugin error

Some users have encountered the following error (or similar ones with slight variations) when attempting to launch the Kilosort4 GUI:
```
QObject::moveToThread: Current thread (0x2a7734988a0) is not the object's thread (0x2a77349d4e0).
Cannot move to target thread (0x2a7734988a0)

qt.qpa.plugin: Could not load the Qt platform plugin "windows" in "<FILEPATH>" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: minimal, offscreen, webgl, windows.
```
This is not specific to Kilosort4, it is a general problem with PyQt (the GUI library we chose to develop with) that doesn't appear to have a single cause or fix. If you encounter this error, please check [issue 597](https://github.com/MouseLand/Kilosort/issues/597) and [issue 613](https://github.com/MouseLand/Kilosort/issues/613) for some suggested solutions.


## Developer instructions

To get the most up-to-date changes to the code, clone the repository and install in editable mode in your Kilosort environment, along with the other installation steps mentioned above. Using the included `environment.yml` to create your environment is recommended, which will also install pytest and the cuda version of Pytorch.
~~~
git clone git@github.com:MouseLand/Kilosort.git
conda env create -f environment.yml
conda activate kilosort
pip install -e Kilosort[gui]
~~~

Then run all tests with:
~~~
pytest --runslow
~~~

To run on GPU:
~~~
pytest --gpu --runslow
~~~

Omitting the `--runslow` flag will only run the faster unit tests, not the slower regression tests.
