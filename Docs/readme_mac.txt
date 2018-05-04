***these instructions have not been tested in a long time, and Matlab appears to have simplified this process quite a bit on other platforms. 
Try running mexGPUall right after installing everything. 
If that fails, the rest of the instructions below might help. Particularly #7 appears required on Mac. ***

Assuming gpu functions have been correctly compiled (see below), a "master_file.m" is available that you should copy to a local path and change for each of your experiments. 
The logic is that the git folder might be updated, and when that happens all extraneous files in that folder will be deleted and any changes you made reverted. 

Mac (instructions from Dan Denman)
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

mexWtW.cu: 93
mexMPreg.cu: 206,233
mexMPsub.cu: 282,287,292
mexMPmuFEAT.cu: 346,356
mexMPregMU.cu: 231,236,264
mexWtW2.cu: 97

8. ran mexGPUall.m

...and MEX completed successfully!
