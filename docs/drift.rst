Drift correction
==============================

Here we get into the details of drift: why it happens, what it looks like, how you can characterize it for your own recordings, and how you can tell if Kilosort is fixing it. The examples in this section are from acute Neuropixels 1.0 recordings in head-fixed mice. 

Why drift happens
------------------------

Some amount of drift is unavoidable. The brain floats inside the skull, and moves when the animal moves, which can be very fast. At slower timescales, physiological changes happen that change the tissue or move it relative to the probe. Some of these physiological changes may be induced by the presence of the probe itself, for example due to an inflammatory response. Thus, there are two main types of drift, which we treat differently: slow (10s of minutes) and fast (10s of seconds). 

Slow drift is not a huge problem if the recording is short (<10 minutes). Fast drift is not a huge problem if the animal is not behaving. However, neither is true in a typical neuroscience experiment: the animal is performing a task which involves some motor actions, and it takes at least 30 minutes to characterize the neural activity. In a typical recording of 1-2 hours, the amount of slow and fast drift we'd expect is comparable, and can be anywhere from 5 to 20um and more if your preparation is unstable. 

What drift looks like
--------------------------

To recognize drift, look for changes in spike amplitude over time, or changes in feature amplitudes, where the feature can be the projection on the principal components of a channel. More dramatically, drift may be recognized by a cluster that "drops out" because all its spikes are lost when the drift is too large. Here is are three examples tracked by Kilosort2, all from the same recording. These clusters appear to have different amounts of drift, which is primarily because they have different sizes: the unit that drifts the most can only be seen on one channel. However, the moments at which drift happens for these units are similar. 

.. image:: https://www.kilosort.org/static/images/amplitudes1.png
   :width: 600

.. image:: https://www.kilosort.org/static/images/amplitudes3.png
   :width: 600

.. image:: https://www.kilosort.org/static/images/amplitudes2.png
   :width: 600

**Slow drift** is generally easier to recognize and fix. A well-known version of slow drift happens after probe insertion in an acute experiment, which is why it is typical to wait 20-30 minutes for the tissue to relax before recording. However, even after that initial phase, there is a smaller amplitude, slow timescale drift that continues for at least a few hours, and has been reported to us even in chronic implants. Cumulative over time, slow drift can have a significant impact on spike sorting. 

**Fast drift** is harder to diagnose but potentially more dangerous, because it may introduce behavior-dependent biases into the data. The bias may be produced if every time an animal performs a certain motor action, the probe moves a little, in which case the spikes from a small neuron may be completely lost. It will then appear as if this neuron was inhibited by movement. Conversely, a neuron may only come into the range of the electrodes during the movement, in which case it will appear as if that neuron is activated by movement. This behavior also makes fast drift harder to diagnose, because most neurons in the brain genuinely have movement-related spiking activity. 

How to diagnose drift
---------------------------

The main way to diagnose drift is, in fact, to run the first step of Kilosort4, which produces a picture of spike times by depth, with the size of the points representing their amplitudes.

.. image:: https://www.kilosort.org/static/images/driftex.PNG
   :width: 800


The spike sorting routine for each batch is a very fast, lite clustering algorithm, where the spikes are first detected as threshold crossings in PCA space, their PCA features are then extracted and the spikes are clustered with scaled k-means. To get enough information from a single batch, there should be many spikes inside that time interval. The default time interval is 2s, which may be too short for <32 channels, in which case we recommend increasing the batch size. 

Drift tracking improves cluster separation
---------------------------------------------------

As a result of drift tracking, units are no longer split into multiple small pieces like in Kilosort1 and other algorithms. Merging back together these small pieces was the most time consuming part of the manual curation in Kilosort1, which is no longer required. On top of the automation benefit, an additional benefit of drift correction is the overall improvement in cluster separation. Why would that be the case? Imagine two neurons that have very similar waveforms but are slightly shifted along the probe. As the probe drifts up, the bottom neuron starts taking the place of the top neuron,  and since they have similar waveforms, it is impossible to tell apart spikes of the bottom neuron at this time from spikes of the top neuron before the drift started. Unless, that is, the spike sorting algorithm is aware of the time in the recording when the different spikes happened. Algorithms like Kilosort2-4, which track units as they drift, maintain a constant separation between two such units, because the separation is only for spikes at a particular drift position. Here is an example from a real recording. 

First, the amplitude timecourses of two very similar units:

.. image:: https://www.kilosort.org/static/images/amplitudes.png
   :width: 600

Now the PCA feature projections across all times, for the two channels where these units are biggest:


.. image:: https://www.kilosort.org/static/images/scatter_PCA.png
   :width: 200


Finally, here are the projections on the time-varying templates of Kilosort2, showing that the units maintain a constant high separation throughout the recording:

.. image:: https://www.kilosort.org/static/images/scatter_TEMP.png
   :width: 300
