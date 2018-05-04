Assuming gpu functions have been correctly compiled (see below), a "master_file.m" is available that you should copy to a local path and change for each of your experiments. The logic is that the git folder might be updated, and when that happens all extraneous files in that folder will be deleted and any changes you made reverted. 

The following are instructions for setting up mex compilation of CUDA files with direct Matlab inputs. Note you need Matlab with the parallel computing toolbox. Instructions below are for Linux and  Windows. Mac instructions are in a separate file (I haven't tested them though). If successful, you should be able to run mexGPUall. 

You should be able to run the code on a CPU without compiling any files, but it will be much, much slower than even on 250$ GPUs. For up to 32 channels, the CPU code might be fast enough. 

Windows

Install Visual Studio Community (2012 or 2013)
Install CUDA 8.0 in Matlab R2017a (and 7.5 in R2016 etc; if in doubt, best to update to latest Matlab and CUDA). If you get a warning of not finding the GPU at the beginning of installation, this might be fine, unless you cannot run mexGPUall, in which case you should try a different combination of Nvidia/CUDA drivers. 

Try to run mexGPUall. If mexcuda gives an error, try the following: copy mex_CUDA_win64.xml (or nvcc_msvc120.xml, or a similarly named file, compatible with your Visual Studio installation version; 11 for 2012 and 12 for 2013) from here
matlabroot/toolbox/distcomp/gpu/extern/src/mex/win64 and into the KiloSort folder (or somewhere in your path). The included file with KiloSort will NOT be compatible with your local environment (unless you start changing paths inside it). 

If your video card is also driving your display, you need to disable the Windows watchdog that kills any process occupying the GPU for too long. 
start regedit,
navigate to HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
create new DWORD key called TdrLevel, set value to 0,
restart PC.

If you find that gpuDevice(1); takes more than 2 minutes each time you 
run then you dont have the space to store its compiled files
GTX 1080 seems to have this issue. 
In order to fix this set an environment variable "CUDA_CACHE_MAXSIZE " on the machine to 
some high value like 1GB.  By default "CUDA_CACHE_MAXSIZE" is 32MB. 
In Windows you can do this in properties > advanced system settings > environment variables. 
In order to set the cache to 1GB use CUDA_CACHE_MAXSIZE 1073741824. 



Linux

UPDATE: for recent video cards/drivers, please see Carl Schoonover's instructions here https://groups.google.com/forum/#!topic/phy-users/g0FSHRI0Nao.

Install CUDA (should ask for a compatible recent version of gcc, will install Nvidia drivers if necessary).

Try to run mexGPUall. If mexcuda gives you an error, try something along the following lines

Append to /home/marius/.bashrc then logout/login:
export CUDA_HOME=/usr/local/cuda-7.5 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
export PATH=${CUDA_HOME}/bin:${PATH} 

Copy mex_CUDA_glnxa64.xml (or a similarly named file, compatible with your Visual Studio installation) from here
matlabroot/toolbox/distcomp/gpu/extern/src/mex/
and into the KiloSort folder (or somewhere in your path). The included file with KiloSort will NOT be compatible with your local environment (unless you start changing paths inside it). 


