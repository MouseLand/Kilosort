.. _gui_guide:

How to use the GUI
==================
This page explains how to use the basic functions of Kilosort4 GUI, in order of the steps you would take to start sorting.


Select data
-----------
Start by selecting a binary data file (.bin, .bat, .dat, or .raw) to load, by clicking on the "Select Binary File" button near the top-left of the GUI. This will open a file dialog that will populate the neighboring text field after a file is selected. You can also paste a filepath directly into the text field.

.. image:: https://www.kilosort.org/static/images/gui_select_binary.png
   :width: 600

Note that binary files should be in row-major (or 'C') order. This is the default for NumPy arrays.


Convert from other formats (optional)
-------------------------------------
If your data is not in one of the supported formats listed in the previous step, you can click the "Convert to Binary" button to open the data conversion tool. Using this tool requires the `SpikeInterface <https://spikeinterface.readthedocs.io/en/latest/>`_ package. To convert your data, you will need to select either a file OR a folder (not both) using the, choose the filetype from the list of supported options, and select the dtype. Then, click "Convert to Binary" (recommended) to copy the data to a new .bin file. Alternatively, you can click "Load As Wrapper" to use the data without copying to a new file, but you will not be able to view results in Phy.

.. image:: https://www.kilosort.org/static/images/gui_convert_data.png
   :width: 600

Note: this tool is only intended to cover the most common formats and options for getting your data into Kilosort. If you don't see your data format or if you run into errors related to extra options that aren't in our GUI, we recommend using SpikeInterface directly to convert your data. Their `documentation here <https://spikeinterface.readthedocs.io/en/latest/modules_gallery/core/plot_1_recording_extractor.html>`_ shows an example of how to save recording traces to .raw format using their own tools.


Choose where to save results
----------------------------
After a binary file is selected, the GUI will automatically populate the text field under "Select Results Dir." with the same path as your binary file, but ending in a "/kilosort4" directory instead of the binary file. If you wanted to change this, you can click the "Select Results Dir." button to open another file dialog, or edit the text field.

.. image:: https://www.kilosort.org/static/images/gui_results_dir.png
   :width: 600


Select a probe
--------------
To select a probe, click the drop-down menu just below "Select Probe Layout." The list will include some default Neuropixels probe layouts. If you've already created your own probe file (.mat, .prb, or .json), you can select "other..." to open a file dialog and navigate to it.

.. image:: https://www.kilosort.org/static/images/gui_select_probe.png
   :width: 600

If you need to create a new probe layout, select "[new]" to open the probe creation tool. Values for 'x-coordinates' and 'y-coordinates' need to be in microns, and can be specified with numpy expressions. For example, a 1-shank linear probe with 4 channels could have `np.ones(4)` in the 'x-coordinates' field instead of `1, 1, 1, 1`. Each field (except name) must have the same number of elements, corresponding to the number of ephys channels in the data. When you are finished setting the values, click "Check" to verify that your inputs are valid. If they are not, an error message will be displayed. Otherwise, the "OK" button will become clickable, which will save the probe to the Kilosort4 probes directory.

.. image:: https://www.kilosort.org/static/images/gui_make_probe.png
   :width: 600

After a probe is selected, you can click "Preview Probe" to see a visualization and verify that the probe geometry looks correct. Checking "True Aspect Ratio" will show a physically proportional representation. Moving the slider will adjust the displayed scale of the contacts.


Load the data
-------------
After you select a probe, the GUI will attempt to automatically determine the correct value for 'number of channels.' Make sure this correctly reflects the number of channels in your datafile, including non-ephys channels. For example, Neuropixels 1 probes output data with 385 channels. Only 384 of those are the ephys data used for sorting, but 'number of channels' should still be set to 385. You may also need to change the dtype of the data (int16 by default) or the sampling rate (30000hz by default). Additionally, you can choose which computing device. By default, the GUI will select the first CUDA GPU detected by PyTorch, or CPU if no GPU is detected.

When you are satisfied with these settings, click "LOAD" at the top left of the GUI to load the data.

.. image:: https://www.kilosort.org/static/images/gui_data_settings.png
   :width: 600


Run spike sorting
-----------------
After loading the data, a heatmap will appear on the right half of the GUI showing a preprocessed version of the data. You can click "raw" at the bottom right to view the data without preprocessing applied. Make sure the data looks like what you expect, including the correct number of seconds along the bottom of the GUI. A common error to look for is diagonal lines in the heatmap, which usually indicates that 'number of channels' does not match the data. When everything looks good, click "Run" near the bottom left to begin spike sorting. When sorting is finished, the results will be saved to the directory indicated under "Select Results Dir."

.. image:: https://www.kilosort.org/static/images/gui_run_sorting.png
   :width: 600

Not pictured: you can now check the "Save Preprocessed Copy" under the "Run" button to save a filtered, whitened, and drift-corrected copy of the data to "temp_wh.dat" in the results directory. This will also reformat the results for Phy so that the preprocessed copy is used instead of the raw binary file.

If you run into errors or the results look strange, you may need to tweak some of the other settings. A handful are shown below 'number of channels' and 'sampling frequency,' or you can click "Extra settings" to open a new window with more options. Mousing over the name of a setting for about half a second will show a description of what the setting does, along with information about which values are allowed. For more detailed suggestions, see :ref:`parameters`

.. image:: https://www.kilosort.org/static/images/gui_extra_settings.png
   :width: 600

If you're still not sure how to proceed, check `issues page on our github <https://github.com/MouseLand/Kilosort/issues>`_ for more help.


Resetting the GUI
-----------------
If the GUI gets stuck on a loading animation or some other odd behavior, try clicking on "Reset GUI" near the top right, which should reset it to the state shown in the first step on this page. If you want to make sure all previous settings are deleted, you can also click "Clear Cache" and then close and re-open the GUI.

.. image:: https://www.kilosort.org/static/images/gui_reset.png
   :width: 600
