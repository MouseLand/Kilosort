*** these instructions have not been tested in a long time, and Matlab appears to have simplified this process quite a bit on other platforms. 
Try running mexGPUall right after installing everything. 
If that fails, the rest of the instructions below might help. Particularly #6 appears required on Mac. 
***

Assuming gpu functions have been correctly compiled (see below), a "master_file.m" is available that you should copy to a local path and change for each of your experiments. 
The logic is that the git folder might be updated, and when that happens all extraneous files in that folder will be deleted and any changes you made reverted. 

Mac (instructions from Dan Denman, updated line numbers in #6. for KS2)
(w/ NVIDIA GeForce GTX 970; OS X 10.13.5, Xcode 9.2, CUDA 9.2)

1. install MATLAB R2017b
2. install parallel computing toolbox license
3. modify ~.bashrc to include:
export MW_NVCC_PATH=/Developer/NVIDIA/CUDA-9.2/bin/nvcc:$MW_NVCC_PATH
export PATH=/Developer/NVIDIA/CUDA-9.2/bin:$PATH
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-9.2/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
NOTE: i am using CUDA 9.2 here, make sure paths and version match your install

4. modify nvcc_clang++.xml which in my MATLAB are here:
/Applications/MATLAB_R2017b.app/toolbox/distcomp/gpu/extern/src/mex/maci64/
changed Version = “7.0” to Version = “9.2” for my CUDA install (lines 92,94,96)
NOTE: i am using CUDA 9.2 here, make sure paths and version match your install

5. Because NVIDIA requires a particular toolkit for particular OS X version (CUDA Toolkit 9.2 for the current 10.13.4+), whereas Matlab links to a different, built in CUDA Toolkit (e.g., CUDA Toolkit 8.0 in Matlab2017b), you need to specify the correct linker libraries when mexcuda tries to compile. this means adding the following:
<-L&quot;/Developer/NVIDIA/CUDAX.X/lib$quot;>, where X.X is the current CUDA Toolkit typically installed in /Developer/NVIDIA/ on OS X
to the LINKERLIBS var on line 51 of the config file in /Applications/MATLAB_R2017b.app/toolbox/distcomp/gpu/extern/src/mex/maci64/nvcc_clang++.xml

when compiling this way, you will also need to copy libcublas.X.X.dylib, libcudart.X.X.dylib, and libcufft.X.X.dylib from /Developer/NVIDIA/CUDAX.X/lib/ to /Applications/MATLAB_R2017b.app/bin/maci64/

6. after that, mexcuda at least finds nvcc. but, errors thrown during mexGPUall.m, like this one:
note: insert an explicit cast to silence this issue
const mwSize dimsu[] = {Nfilt, Nfilt, ((2 * nt0) ­ 1)};
^~~~~
static_cast<mwSize>( )

Right, so i inserted static_cast<mwSize>( ) for named variables wherever necessary. Here is an example of the diff results from from mexClustering2.cu, showing the new lines:

line:292:   const mwSize ddWU[] 	= {static_cast<mwSize>(NrankPC) * static_cast<mwSize>(Nchan), static_cast<mwSize>(Nfilters)};
line:351:   const mwSize dimst[]      = {static_cast<mwSize>(Nspikes),1};  
line:352:   const mwSize dimst2[] 	= {static_cast<mwSize>(Nspikes),static_cast<mwSize>(Nfilters)};  
line:353:   const mwSize dimst4[] 	= {static_cast<mwSize>(Nfilters),1};  

which was in these places:
file              : line numbers
mexClustering.cu  : 292,351,352,353
mexDistances2.cu  : 160
mexFilterPCs.cu   : 102
mexGetSpikes2.cu  : 320,379
mexMPnu8.cu       : 608,611,733,743,752
mexSVDsmall2.cu   : 319
mexThSpkPC.cu     : 267,272
mexGetSpikes.cu   : 309
mexWtW2.cu        : 113

7. run mexGPUall.m

...and MEX should complete successfully! 
