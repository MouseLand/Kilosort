.. _parameters:

When to adjust default settings
===============================
This page will give suggestions for when to change specific settings and why. Not every configurable setting will be covered here, but all settings include a short description in `kilosort/parameters.py <https://github.com/MouseLand/Kilosort/blob/main/kilosort/parameters.py>`_. The same description can also be seen in the GUI by mousing over the name of a setting for about half a second.


``n_chan_bin`` (number of channels)
-----------------------------------
This should reflect the total number of channels in the binary file, `including non-ephys channels not used for sorting`. If you load your data in the GUI and see repeating diagonal patterns in the data, you probably need to change this value.


``batch_size``
--------------
This sets the number of samples included in each batch of data to be sorted, with a default of 60000 corresponding to 2 seconds for a sampling rate of 30000. For probes with fewer channels (say, 64 or less), increasing ``batch_size`` to include more data may improve results because it allows for better drift estimation (more spikes to estimate drift from). 


``nblocks``
-----------
This is the number of sections the probe is divided into when performing drift correction. The default of ``nblocks = 1`` indicates rigid registration (the same amount of drift is applied to the entire probe). If you see different amounts of drift in your data depending on depth along the probe, increasing ``nblocks`` will help get a better drift estimate. ``nblocks=5`` can be a good choice for single-shank Neuropixels probes. For probes with fewer channels (around 64 or less) or with sparser spacing (around 50um or more between contacts), drift estimates are not likely to be accurate, so drift correction should be skipped by setting ``nblocks = 0``.


``Th_universal`` and ``Th_learned``
-----------------------------------
These control the threshold for spike detection when applying the universal and learned templates, respectively (loosely similar to Th(1) and Th(2) in previous versions). If few spikes are detected, or if you see neurons disappearing and reappearing over time when viewing results in Phy, it may help to decrease ``Th_learned``. To detect more units overall, it may help to reduce ``Th_universal``. Try reducing each threshold by 1 or 2 at a time.


``tmin`` and ``tmax``
---------------------
This sets the start and end of data used for sorting (in seconds). By default, all data is included. If your data contains recording artifacts near the beginning or end of the session, you can adjust these to omit that data. "inf" and "np.inf" can be used for tmax to indicate end of session in the GUI and API respectively. 


``nt``
------
This is the number of time samples used to represent spike waveforms, as well as the amount of symmetric padding for filtering. The default represents 2ms + 1 bin for a sampling rate of 30kHz. For a different sampling rate, you may want to adjust accordingly. For example, ``nt = 81`` would be the 2ms equivalent for 40kHz.


``dmin`` and ``dminx``
----------------------
These adjust the vertical and lateral spacing, respectively, of the universal templates used during spike detection, as well as the vertical and lateral sizes of channel neighborhoods used for clustering. By default, Kilosort will attempt to determine a good value for ``dmin`` based on the median distance between contacts, which tends to work well for Neuropixels-like probes. However, if contacts are irregularly spaced, you may need to specify this manually. The default for ``dminx`` is 32um, which is also well suited to Neuropixels probes. For other probes, try setting ``dminx`` to the median lateral distance between contacts as a starting point.

Note that as of version 4.0.11, the ``kcoords`` variable in the probe layout will be used to restrict template placement within each shank. Each shank should have a unique ``kcoords`` value that is the same for all contacts on that shank.

``min_template_size``
---------------------
This sets the standard deviation of the smallest Gaussian spatial envelope used to generate universal templates, with a default of 10 microns. You may need to increase this for probes with wider spaces between contacts.


``nearest_chans`` and ``nearest_templates``
-------------------------------------------
This is the number of nearest channels and template locations, respectively, used when assigning templates to spikes during spike detection. ``nearest_chans`` cannot be larger than the total number of channels on the probe, so it will need to be reduced for probes with less than 10 channels. ``nearest_templates`` does not have this restriction. However, for probes with around 64 channels or less and sparsely spaced contacts, decreasing ``nearest_templates`` to be less than or equal to the number of channels helps avoid numerical instability.


``x_centers``
-------------
The number of x-positions to use when determining centers for template groupings. Specifically, this is the number of centroids to look for when using k-means to cluster the x-positions for the probe. In most cases you should not need to specify this. However, **for probes with contacts arranged in a 2D grid**, we recommend setting ``x_centers`` such that centers are placed every 200-300um so that there are not too many templates in each group. For example, for an array that is 2000um in width, try ``x_centers = 10``. If contacts are very densely spaced, you may need to use a higher value for better performance.


``duplicate_spike_ms``
------------------------
After sorting has finished, spikes that occur within this many ms of each other, from the same unit, are assumed to be artifacts and removed. If you see otherwise good neurons with large peaks around 0ms when viewing correlograms in Phy, increasing this value can help remove those artifacts.

**Warning!!!** Do not increase this value beyond 0.5ms as it will interfere with the ACG and CCG refractory period estimations (which normally ignores the central 1ms of the correlogram).
