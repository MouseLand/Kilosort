*** these instructions have not been tested in a long time, and Matlab appears to have simplified this process quite a bit on other platforms. 
Try running mexGPUall right after installing everything. 
If that fails, the rest of the instructions below might help. Particularly #7 appears required on Mac. 
***

Assuming gpu functions have been correctly compiled (see below), a "master_file.m" is available that you should copy to a local path and change for each of your experiments. 
The logic is that the git folder might be updated, and when that happens all extraneous files in that folder will be deleted and any changes you made reverted. 

Mac (instructions from Dan Denman, updated line numbers in 7 for KS2)
(w/ NVIDIA GeForce GTX 970; OS X 10.11.3, Xcode 7.2.1, CUDA 7.5.21)

1. installed MATLAB R2015b
2. installed parallel computing toolbox license
3. modified ~.bashrc to include:
export CUDA_HOME=/usr/local/cuda
export PATH=/Developer/NVIDIA/CUDA­7.5/bin:$PATH
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA­7.5/lib:$DYLD_LIBRARY_PATH
4. tried to run mexGPUall.m, MATLAB couldn’t find a compiler
updated Xcode to 7.2.XXX
5. did a bunch more stuff trying to specify MATLAB compiler, including adding this to ~.bashrc. not sure any of this is actually necessary.

export CUDA_PATH=/usr/local/cuda
export MW_NVCC_PATH=/Developer/NVIDIA/CUDA­7.5/bin/nvcc:$MW_NVCC_PATH

6. eventually, i found this thread, which led me to switch the ‘mex’ commands in mexGPUall.m to ‘mexcuda’
I also had to modify nvcc_clang++.xml, which in my MATLAB is here:
/Applications/MATLAB_R2015b.app/toolbox/distcomp/gpu/extern/src/mex/maci64/nvcc_clang++.xml
changed Version = “7.0” to Version = “7.5” for my CUDA install
add some paths so it could find my Xcode install.
I also did this to nvcc_clang++_dynamic.xml. if you ever want more details I can provide.

7. after that, mexcuda at least found nvcc. but, errors thrown during mexGPUall.m, like this one:

/Users/administrator/Documents/MATLAB/KiloSort­master/mexWtW.cu:93:32: note: insert an
explicit cast to silence this issue
const mwSize dimsu[] = {Nfilt, Nfilt, ((2 * nt0) ­ 1)};
^~~~~
static_cast<mwSize>( )

Right, so i inserted static_cast<mwSize>( ) wherever necessary, which was in these places:

file : line #s

mexGetSpikes.cu   309
mexMPnu8.cu        684, 803, 811, 822, 826
mexSVDsmall2.cu 316  [also changed ‘1.0f’ to ‘1.0’ in line 256]
mexWtW2.cu         113
mexFilterPCs.cu    102
mexClustering.cu   222, 223, 224
mexDistances.cu   149

8. ran mexGPUall.m

*** update 6/28/18. Version compatibility between Mac OS / CUDA Toolkit / Matlab:
9. Because NVIDIA requires a particular toolkit for particular OS X version (CUDA Toolkit 9.2 for the current 10.13.4+), whereas Matlab links to a different, built in CUDA Toolkit (e.g., CUDA Toolkit 8.0 in Matlab2017b), you need to specify the correct linker libraries when mexcuda tries to compile. this means adding the following:
<-L&quot;/Developer/NVIDIA/CUDAX.X/lib$quot;>, where X.X is the current CUDA Toolkit typically installed in /Developer/NVIDIA/
to the LINKERLIBS var on line 51 of the config file in /Applications/MATLAB_R2017b.app/toolbox/distcomp/gpu/extern/src/mex/maci64/nvcc_clang++.xml

when compiling this way, you will also need to copy libcublas.X.X.dylib, libcudart.X.X.dylib, and libcufft.X.X.dylib from /Developer/NVIDIA/CUDAX.X/lib/ to /Applications/MATLAB_R2017b.app/bin/maci64/

***

...and MEX completed successfully!
