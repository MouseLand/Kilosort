Running Kilosort
================

Launching the GUI
-----------------
1. Open the GUI using :code:`python -m kilosort`
2. Select the path to the binary file and optionally the results directory.
   We recommend putting the binary file on an SSD for faster processing.
3. Select the probe configuration (mat files recommended, they actually exclude
   off channels unlike prb files)
4. Hit :code:`LOAD`. The data should now be visible.
5. Hit :code:`Run`. This will run the pipeline and output the results in a
   format compatible with Phy, the most popular spike sorting curating software.


Using the API
-------------
In brief:
::
   
   from kilosort import run_kilosort
   results = run_kilosort()

See the "Step-by-step" tutorial for a more detailed walkthrough.


Integration with Phy GUI
------------------------
`Phy <https://github.com/kwikteam/phy>`_ provides a manual clustering interface for refining the results of the
algorithm. Kilosort4 automatically sets the "good" units in Phy based on a
<10% estimated contamination rate with spikes from other neurons (computed from
the refractory period violations relative to expected).
|
Check out the `Phy <https://github.com/kwikteam/phy>`_ repo for full install
instructions, but briefly:
::

    pip install phy --pre --upgrade

This should work fine in the :code:`kilosort` environment you made (if not make
a new environment for phy). Then change to the results directory from
:code:`kilosort4` (by default a folder named kilosort4 in the binary directory)
and run:

::

    phy template-gui params.py
