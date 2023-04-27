Installing Kilosort
===================
You can run Kilosort4 without installing it locally using google colab.
An `example colab notebook <https://colab.research.google.com/drive/1gFZa8TEBDXmg_CB5RwuT_52Apl3hP0Ie?usp=sharing>`_
is available. It will download some test data, run kilosort4 on it,
and show some plots.


Installation
------------
1. Preparing your system
^^^^^^^^^^^^^^^^^^^^^^^^
If you have an older kilosort environment you can remove it with
:code:`conda env remove -n kilosort` before creating a new one.

Also make sure the NVIDIA driver for your GPU is installed, and the CUDA and
CUDNN libraries for it are installed. We recommend CUDA 11.7.


2. Install `Anaconda <https://www.anaconda.com/products/distribution>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the `conda` command is unrecognized in subsequent steps, you may need to
add Anaconda to your system path using the Anaconda prompt.

Note that this step is optional, but recommended if you aren't very familiar
with the ins and outs of Python environment management.


3. Create and activate environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open a terminal or Anaconda prompt and type:
::

    conda create --name kilosort python=3.8

Python version 3.8 or 3.9 is recommended, but 3.10 should work as well.
After the environment is created, activate it with:
::

    conda activate kilosort


4. Install Kilosort from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Download the code repository with git, and change directory to the root directory
of Kilosort repository. Then run
::

    python -m pip install .

Optionally, you can add the [gui] tag to install the GUI as well:
::

    python -m pip install .[gui]


5. Install the GPU version of pytorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This will install the CPU version of pytorch. To fix this, first run
::

    pip uninstall torch

Then follow the instructions `here <https://pytorch.org/get-started/locally/>`_
to determine what version to install. The Anaconda install is strongly recommended,
and then choose the CUDA version that is supported by your GPU (newer GPUs may
need newer CUDA versions > 10.2). For instance this command will install the 11.7
version on Linux and Windows (note the :code:`torchvision` and :code:`torchaudio`
commands are removed because kilosort doesn't require them):
::

    conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia

Note you will always have to run :code:`conda activate kilosort` before you run
Kilosort. If you want to run jupyter notebooks in this environment, then also
run :code:`conda install jupyter` and :code:`python -m pip install matplotlib`.


For developers
--------------
Install pytest to verify that the installation is working properly
::

    pip install pytest
    python -m pytest --runslow


This process takes several minutes since currently we run through the full
spike-sorting pipeline on sample data and compare the output to pre-computed
results. The first time you run tests, the sample dataset will be downloaded
(~1GB), so the tests will take a little extra time.
