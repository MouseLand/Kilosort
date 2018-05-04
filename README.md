# Fast spike sorting for hundreds of channels #

Implements an integrated template matching framework for detecting and clustering spikes from multi-channel electrophysiological recordings. Very fast when a GPU is available, but can also run on the CPU side. Described in this NIPS paper

Pachitariu M, Steinmetz NA, Kadir S, Carandini M and Harris KD (2016). Fast and accurate spike sorting of high-channel count probes with Kilosort. Advances In Neural Information Processing Systems. 4448-4456
https://papers.nips.cc/paper/6326-fast-and-accurate-spike-sorting-of-high-channel-count-probes-with-kilosort

The following preprint contains slightly different information:

Pachitariu M, Steinmetz NA, Kadir S, Carandini M and Harris KD (2016). Kilosort: realtime spike-sorting for extracellular electrophysiology with hundreds of channels. 
bioRxiv dx.doi.org/10.1101/061481, [link](http://biorxiv.org/content/early/2016/06/30/061481). 

### Installation ###
If you are running on the GPU, you must run mexGPUall in the CUDA folder after setting up mexcuda in Matlab. Steps for achieving this in Windows 10 are below; see the Docs folder for more details and other operating systems.

#### Windows 10 installation ####

1. Install matlab R2018a from www.mathworks.com. You can use older versions of matlab but see notes below. 
2. Install cuda 9.0 from https://developer.nvidia.com/cuda-90-download-archive. This version of CUDA works with Matlab R2018a. For older versions of matlab, skip this for now, and at a later stage it will tell you what version of CUDA you need. Then search on google or the NVIDIA site for that version.
3. Install visual studio 2013 community from https://www.visualstudio.com/vs/older-downloads/. Installing this should come after installing matlab or else matlab may not find the compiler, in which case you won't see it in the mex setup list (see below). You should be able to install again on top of an old installation, if you already had it from before matlab installation. 
4. Now in matlab, choose the compiler:.
```matlab
>> mex -setup C++ % gives an output like below:

To choose a different C++ compiler, select one from the following:
MinGW64 Compiler (C++)  mex -setup:'C:\Program Files\MATLAB\R2018a\bin\win64\mexopts\mingw64_g++.xml' C++
MinGW64 Compiler with Windows 10 SDK or later (C++)  mex -setup:'C:\Program Files\MATLAB\R2018a\bin\win64\mexopts\mingw64_g++_sdk10+.xml' C++
Microsoft Visual C++ 2013  mex -setup:'C:\Program Files\MATLAB\R2018a\bin\win64\mexopts\msvcpp2013.xml' C++

% you want to pick the 2013 edition, so copy and paste the corresponding instruction. For me that's: 
>> mex -setup:'C:\Program Files\MATLAB\R2018a\bin\win64\mexopts\msvcpp2013.xml' C++
```
5. Compile kilosort
```matlab
>> cd C:\my\github\directory\KiloSort\CUDA % your kilosort repository directory
>> mexGPUall % should get a bunch of warnings plus "MEX completed successfully", several times. 
% at this stage it may tell you that you need a different version of cuda for older matlab installations. 
```

For more about mexcuda installation, see these ([instructions](http://uk.mathworks.com/help/distcomp/mexcuda.html)). 

### Verifying installation and test run ###

You can verify that the code has been installed correctly by running master_eMouse inside the eMouse folder. See first readme_eMouse.txt. You can also use these scripts to understand how to pass the right settings into Kilosort (will depend on your probe, channel map configuration etc), and what you should be seeing in Phy during manual cleanup of Kilosort results. There are many parameters of the simulation which you can tweak to make it harder or easier, and perhaps more similar to your own data. 

### General instructions for running Kilosort ###

1. Make a copy of master_file_example_MOVEME.m and \configFiles\StandardConfig_MOVEME.m and put them in the directory with your data. 
2. Generate a channel map file for your probe using \configFiles\createChannelMap.m as a starting point. 
3. Edit the config file with desired parameters. You should at least set the file paths (ops.fbinary, ops.fproc (this file will not exist yet - kilosort will create it), and ops.root), the sampling frequency (ops.fs), the number of channels in the file (ops.NchanTOT), the number of channels to be included in the sorting (ops.Nchan), the number of templates you want kilosort to produce (ops.Nfilt), and the location of your channel map file (ops.chanMap). 
4. Edit master_file so that the paths at the top (lines 3-4) point to your local copies of those github repositories, and so that the configuration file is correctly specified (lines 6-7). 

To understand the parameters that can be adjusted in Kilosort, please refer to the example configuration files. The description of each parameter is inline with its assigned (default) setting, which you can change. 

### Integration with Phy GUI ###
Kilosort provides a results file called "rez", where the first column of rez.st are the spike times and the second column are the cluster identities. However, the best way to use this software is together with [Phy](https://github.com/kwikteam/phy), which provides a manual clustering interface for refining the results of the algorithm. 

NOTE that you need to use a special branch of Phy with Kilosort. Instructions in Docs/phy_installation_with_templates.txt 

You also need to install [npy-matlab](https://github.com/kwikteam/npy-matlab), to provide read/write functions from Matlab to Python, because Phy is written in Python.

Detailed instructions for interpreting results are provided [here](https://github.com/kwikteam/phy-contrib/blob/master/docs/template-gui.md).

NOTE that if you run the auto-merge feature ("merge_posthoc2"), then the fifth column of rez.st contains the final cluster identities. This grouping is then exported to Phy and interpreted the same way a manual merge is interpreted. This means that you can still use Phy's functionality to break the cluster back up. 

### Matlab output structures ###

Kilosort is best used in conjunction with Phy. The .npy and .csv output files can then be loaded back into Matlab, following these general instructions: https://github.com/kwikteam/phy-contrib/blob/master/docs/template-gui.md. For a full example, see the tutorial with Neuropixels results data available here: http://data.cortexlab.net/singlePhase3/ and here: http://data.cortexlab.net/dualPhase3/. 

However, in some situations you might need to use the Matlab results structures. Here is an explanation of these variables, available inside the struct called "rez" 

xc, yc: x and y coordinates of each channel on the probe, in the order of channels provided in the channel map (default is linear, 1:1:nChannels). 

connected: whether a channel in the original binary dat is "connected", or "active". Inactive channels are ignored.

Wrot: cross-channel whitening matrix. Wrot * high_pass_filtered_data = post_data, where post_data is the postprocessed data on which the Kilosort algorithm is applied. 

WrotInv: is the matrix inverse of Wrot. WrotInv * post_data = high_pass_filtered_data

ops: keeps all the configuration settings provided by the user, and cumulative information added throghout the Kilosort steps. 

st: first column is the spike time in samples, second column is the spike template, third column is the extracted amplitude, and fifth column is the post auto-merge cluster (if you run the auto-merger). 

mu: mean amplitude for each template

U: low-rank components of the spatial masks for each template

W: low-rank components of the temporal masks for each template

dWU: average of a subset of spikes corresponding to each template. The low-rank decomposition of this matrix results in W and U. 

Wraw: the spike template, un-whitened by the operation Wraw(:,:,n) = Wrotinv' * (U(:,n,:) * W(:,n,:)'), for each template n. 

simScore: correlation between all pairs of templates.

cProj: projections of each detected spike onto the principal components of the channels corresponding to the spike's assigned template. The channel order for each template is available in iNeigh.

iNeigh: for each template, the channels with largest amplitudes are indexed in order (default 12). This indexing is used to sort coefficients in cProj. Notice this is a fundamentally sparse scheme: only the top channels for each template are stored. 

cProjPC: projections of each detected spike onto the top templates most similar to the spike's assigned template. The nearest-template order for each template is available in iNeighPC.

iNeighPC: for each template, the other templates with largest similarity are indexed in order (default 12). This indexing is used to sort coefficients in cProjPC. Notice this is a fundamentally sparse scheme: only the top closest template for each template are stored. 


### Questions ###

Please create an issue for bugs / installation problems. 

### Licence ###

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

