### eMouse simulator with drift ###

![](https://github.com/MouseLand/Kilosort2/blob/master/Docs/img/simulation_drift.png)


This simulator is an adaptation of the original eMouse simulator that accompanied Kilosort 1. The current version adds simulation of probe drift. The waveforms at arbitrary points on the probe are estimated using simple interpolation of measured waveforms from "oversampled" measurments. Waveforms for small neurons in this example are drawn from a striatum recording in the Kampff lab Ultradense Survey [1], analyzed by Susu Chen and Nick Steinmetz [2]. Waveforms for large neurons are taken from recordings in midbrain peformed by Susu Chen using an imec 3A probe.

The latest version adds the ability to model the noise on a real data set, by matching the frequency spectrum and cross correlation between sites. See comments in make_eMouseData_drift.m for more detail.

For more information about how to run the simulator, go to the [eMouse wiki page](https://github.com/MouseLand/Kilosort2/wiki/4.-eMouse-simulator-with-drift).

Built by Jennifer Colonell. 


[1] Ultra dense survey http://www.kampff-lab.org/ultra-dense-survey/  
[2] S. Chen, J.P. Neto, M. Pachitariu, A. R. Kampff, N. A. Steinmetz, ‘On the shape and extent of extracellular action potential waveforms across the rodent brain’. SfN 2018. 



