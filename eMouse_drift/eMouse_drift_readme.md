### eMouse simulator with drift ###

This simulator is an adaptation of the original eMouse simulator that accompanied Kilosort 1. The current version adds simulation of tissue drift. The waveforms at arbitrary points on the probe are estimted using simple linear interpolation of measured waveforms from "oversampled" measurments. Waveforms for small neurons in this example are drawn from a striatum recording in the Kampff lab Ultradense Survey, analyzed by Susu Chen and Nick Steinmetz. Waveforms for large neurons are taken from recordings in midbrain peformed by Susu Chen using an imec 3A probe.

### Running the model with defaults ###

The master_eMouse_drift script will generate a 1200 second recording with moderately dense units (~1/site) and significant drift (20 um). To run the script, set the paths for the data directory, temporary directory, and the location of the Kilosort2 code; also set 'useParPool=0' if your workstation has <10 cores. After the data is created, Kilosort2 is called to sort the data, and the sorting results are compared to the known spike times to assess the performance of the sort. The final table written out to the command window gives the details for each ground truth unit, including the cluster labels that appear in Phy, so you can see what merges you'd make with ideal manual curation.

The amplitudes in this example have been made uniform and the "drift" is rigid uniform motion of units in a perfect sine wave. This makes a good example for seeing how much drift can be compensated before units are lost. To make data that is a closer model for your own data, adjust paramters at the start of make_eMouse_drift.m.

### Model limitations ###

Some important differences between the model and real data include:
- Spikes are uniformly distributed in time
- Units are uniformly distributed in space
- No variation of the waveform except that due to movement of the units
- Simple gaussian noise with low correlation between sites

Real data with episodic firing of neurons and soma concentrated in layers will generally be more difficult to sort.


### Some details of the model ###

The firing rate for each unit is drawn from a uniform distribution with limits set by the parameter `fr_bounds`.
A set of time differences bewteen spikes is created using random numbers from a geometric distribution with with probability calculated from the firing rate. Any time differences < than 2 ms (appropriate refractory time) are removed from the set, and the spike times are calculated from the cumulative sums of the time differences.

The input waveforms are averages from the data sets mentioned above. For each instance of a spike by a unit, the waveforms are calculated for each site given the current position of the unit. The amplitude for the current spike is drawn from a gamma distribution with width set by the `amp_std` parameter. This creates a gaussian-like distribution but with no negative values.

The noise is gaussian distributed with rms set by `rms_noise`. It is then smoothed in time and space by convolving with gaussians of width `tsmooth` and `chsmooth`. Increasing these will create more correlation (which may be a closer match to real data) but will also change the frequency of the noise in the raw data.

The drift can be either uniform over the set of units or "pinned" at a point specified by the user. The motion may be sinusoidal or a step followed by exponential decay. The parmeters are explained in detail in the code comments.






