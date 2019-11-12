# Intro
This is a simple batch tensor addition example with Xilinx SDAccel and CMake.  
The targeted SDx version is 2019.1 but it also should work fine with 2018.3 without any modifications.

# Steps to...
The path to SDAccel platform file(*.xpfm) should be set to the environment variable `AWS_PLATFORM`.  
For configuring CMake and building the host program, run:
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
# Useful Links
[Link1](https://github.com/Xilinx/SDAccel-Tutorials/blob/master/docs/Pathway3/ProfileAndTraceReports.md)  
[Link2](https://github.com/Xilinx/SDAccel-Tutorials/blob/master/docs/Pathway3/HardwareExec.md)  
# Libraries
[HLS Lib](https://github.com/definelicht/hlslib)
[SDAccel Development Environment Help for 2019.1](https://www.xilinx.com/html_docs/xilinx2019_1/sdaccel_doc/znf1520531165689.html#znf1520531165689)  
[SDx Command and Utility Reference Guide for 2019.1](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2019_1/ug1279-sdx-command-utility-reference-guide.pdf)  