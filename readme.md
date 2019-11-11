# Intro
This is a simple batch tensor addition example with Xilinx SDAccel and CMake.  
Targeted SDx version is 2019.1 but it also should work fine with 2018.3 without any modifications.

# Steps to...
The path to SDAccel platform file(*.xpfm) should be set to the environment variable `AWS_PLATFORM`.  
For configuring CMake and building host program, run:
```
mkdir build
cmake ..
make
```
then,
##### For Software Emulation:
```
make compile_swemu
make link_swemu
sh launch_swemu.sh
```
##### For Hardware Emulation:
```
make compile_hwemu
make link_hwemu
sh launch_hwemu.sh
```
##### For Hardware Execution(Real FPGA):
```
make compile_hw
make link_hw
sudo sh
source $XILINX_XRT/setup.sh
./MyHostExecutable
```