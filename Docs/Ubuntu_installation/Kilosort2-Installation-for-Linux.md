### Tested system specs (Mari Sosa, March 2020):
* OS: Ubuntu 18.04.2
* Graphics card: NVIDIA GeForce RTX 2070 Super, 8 GB
* Graphics driver: started with none (see below); installed nvidia 440.64 with CUDA
* Linux kernel: 5.3.0-40-generic
* MATLAB version: R2019a
* CUDA version: 10.0

`$` indicates command to enter in the linux terminal \
`>>` indicates command to enter in Matlab

## CUDA installation

1. Check compatibility of your system following [NVIDIA CUDA pre-installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#abstract). **Note that Linux kernels newer than 4.15.0 work just fine**. 
2. Purge (uninstall) previously installed nvidia drivers and any previously installed cuda (step 2.7 in the above guide), and reboot.  This should not break your graphics, because Ubuntu distributions come with an open-source video driver called nouveau that will be used on reboot:
```
$ sudo apt purge nvidia-*
$ sudo apt purge cuda*
$ sudo apt autoremove
$ sudo reboot
```
3. Ubuntu should now be using your default nouveau drivers; you can check this with:
```
$ lshw -c display
```
  * configuration line should say `driver-nouveau`
  * If for some reason your graphics are broken, see Troubleshooting.
4. Determine what CUDA version you need to install to match your Matlab version from [this list](https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html;jsessionid=9b9a3c1f879079e0c083ef7d0c5f).  Example: CUDA 10.0 for Matlab 19a.
5. Download the version of your [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive), using the network option. 
    1. [Direct link for downloading v10.0](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork)
    2. Select `Linux`, `x86_64`, `Ubuntu`, `18.04`, `deb(network)`; assumes network connectivity.
    3. Download the .deb file (to anywhere, presumably ~/Downloads).
    4. Follow the NVIDIA installation instructions almost exactly, modifying the last line to specify the cuda version you want to install, as follows:
```
$ cd ~/Downloads
$ sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get install cuda-10-0
```
5. Correct output should look like [this](https://github.com/mari-sosa/Kilosort2/blob/master/Docs/Ubuntu_installation/cuda-10-0_install_output.txt): 
6. Reboot, then check that cuda has installed, and that you’re running the nvidia graphics driver now.
```
$ ls -l /usr/local
```
  * You should see a directory for your cuda version and a symbolic link to that directory called `cuda`
  * Use either of the following to check your nvidia driver:
```
$ nvidia-smi
$ cat /proc/driver/nvidia/version
```
  * should show that you’re running the nvidia driver version that installed with cuda

## Matlab installation
1. Install Matlab if you haven't already and launch it.

## Kilosort setup 
1. Check that the CUDA toolkit version is correctly recognized by Matlab:
```
>> gpuDevice
```
  * `ToolkitVersion` should say `10`.  It’s ok if `DriverVersion` says something different, like `10.1` or `10.2`.
2. Clone the Kilosort2 repo (or your own forked version) to your local source code directory:
```
$ cd ~/Src
$ git clone https://github.com/MouseLand/Kilosort2.git
```
3. In Matlab, add the Kilosort2 directory to your path and go to the CUDA subdirectory. Run mexGPUall.m.
```
>> cd ~/Src/Kilosort2/CUDA
>> mexGPUall
```
4. Clone the npy-matlab repo and add it to your Matlab path, so that you have the necessary functions to save phy output.
```
$ git clone https://github.com/kwikteam/npy-matlab.git
```
5. CUDA and Kilosort should be ready to use.  

***

# Troubleshooting
**If your graphics don't work on reboot** (e.g. you get the "black screen of death"), try the following.  Keep in mind that there may be different causes for this problem, so this is not guaranteed to help.
* Hard reboot -- press and hold the power button until the computer turns off, then turn it back on.
* At the first splash screen, press ESC repeatedly until the GRUB boot screen comes up.
* Select "Advanced Ubuntu Options" (arrow down and press Enter).
* Select your kernel with "recovery mode".
* In recovery mode, first "Enable Networking". This will take a couple seconds.
* When the recovery mode menu comes back up, select "Drop down to root shell." **Be careful, everything you do here is as root and will not require a password.**
* Check if you still have `ubuntu-desktop` and the `xorg` server.
```
$ dpkg -l ubuntu-desktop
$ dpkg -l xorg
```
* If this returns something like "none" for the version or "package not found," install the package and reboot.
```
$ apt install ubuntu-desktop
$ apt install xorg
$ reboot
```

  \
**If your graphics don't work specifically after installing CUDA**, enter recovery mode as above and check that the nvidia driver built correctly:
```
$ dkms status
```
* Should return (for the system specs listed above): `nvidia, 440.64.00, 5.3.0-40-generic, x86_64: installed`
* If it does not say "installed" try the following to register, build, and install the driver:
```
$ dkms add -m nvidia -v 440.64
$ dkms build -m nvidia -v 440.64
$ dkms install -m nvidia -v 440.64
$ reboot
```
* At this point, if things are still broken, try starting over with the nvidia and cuda purge.
