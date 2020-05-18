# Kilosort2: automated spike sorting with drift tracking and template matching on GPUs #

![](https://github.com/MouseLand/Kilosort2/blob/master/Docs/img/templates.png)

Welcome to Kilosort2, our software for spike sorting electrophysiological data up to 1024 channels. In many cases, and especially for Neuropixels probes, the automated output of Kilosort2 requires minimal manual curation.

There are currently two implementations of Kilosort2, the original matlab version and a python port. **Only the matlab version is stable right now**. We expect the python version to become stable over the next few months, and probably preferable after that. Please post in an issue if you'd like to help out with it.

There is currently no preprint or paper for Kilosort2, so please read the wiki to find out [how it works](https://github.com/MouseLand/Kilosort2/wiki), and especially the [drift correction](https://github.com/MouseLand/Kilosort2/wiki/3.-More-on-drift-correction) section. Kilosort2 improves on Kilosort primarily by employing drift correction, which changes the templates continuously as a function of drift. Drift correction does not depend on a particular probe geometry, but denser spacing of sites generally helps to better track neurons, especially if the probe movement is large. Kilosort2 has been primarily developed on awake, head-fixed recordings from Neuropixels 1.0 data, but has also been tested in a few other configurations. To get a sense of how probe drift affects spike sorting, check out our "eMouse" simulation [here](https://github.com/MouseLand/Kilosort2/tree/master/matlab/eMouse_drift) and [its wiki page](https://github.com/MouseLand/Kilosort2/wiki/4.-eMouse-simulator-with-drift).

(MATLAB version only) To aid in setting up a Kilosort2 run on your own probe configuration, we have developed a [graphical user interface](https://github.com/MouseLand/Kilosort2/wiki/1.-The-GUI) where filepaths can be set and data loaded and visually inspected, to make sure Kilosort2 sees it correctly. The picture above is another GUI visualization: it shows the templates detected by Kilosort2 over a 60ms interval from a Neuropixels recording.

(Both versions) The final output of Kilosort2 can be visualized and curated in the [Phy GUI](https://github.com/kwikteam/phy), which must be installed separately (we recommend the development version). Since Phy is in Python, you will also need the [npy-matlab ](https://github.com/kwikteam/npy-matlab) package. 

### Questions

Please create an issue for bugs / installation problems.

### Licence 

#### Matlab Version
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
