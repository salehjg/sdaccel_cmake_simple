cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)

PROJECT(SDAccel_CMake_Simple)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)
find_package(SDAccel REQUIRED)

add_executable(MyHostExecutable src/MyHostExecutable.cpp src/xcl2.cpp)
include_directories(${SDAccel_INCLUDE_DIRS} inc/)
target_link_libraries(MyHostExecutable ${SDAccel_LIBRARIES})

set(PASTEBIN_0_1 "0" CACHE STRING "Enable or disable PasteBin agent for uploading autobuild log files to PasteBin.") 
set(PASTEBIN_USERNAME "" CACHE STRING "PasteBin Account's Username") 
set(PASTEBIN_PASSWORD "" CACHE STRING "PasteBin Account's Password") 
set(PASTEBIN_API_KEY  "" CACHE STRING "PasteBin Account's API Key") 

# Creates TCL scripts for the given kernel.
function(synthesis_target
        KERNEL_FILE
        KERNEL_NAME
        KERNEL_CLOCK
        KERNEL_CLOCK_UNCERTAINTY
        HLSLIB_PART_NAME
        SYNTHESIS_FLAGS)

    set(HLSLIB_PROJECT_NAME "${KERNEL_NAME}")
    set(HLSLIB_SRC_SYNTHESIS "${KERNEL_FILE}")
    set(HLSLIB_ENTRY_FUNCTION "${KERNEL_NAME}")
    set(HLSLIB_TARGET_CLOCK "${KERNEL_CLOCK}")
    set(HLSLIB_TARGET_CLOCK_UNCERTAINTY "${KERNEL_CLOCK_UNCERTAINTY}")
    set(HLSLIB_SYNTHESIS_FLAGS ${SYNTHESIS_FLAGS})
    configure_file(scripts/Synthesis.tcl.in Synthesize${KERNEL_NAME}.tcl)

endfunction()

configure_file(scripts/SynthesisScriptLauncherBatch.sh.in SynthesisScriptLauncherBatch.sh)
add_custom_target(synthesis COMMAND sh SynthesisScriptLauncherBatch.sh)

# Creating required tcl scripts to run hls synthesis
synthesis_target("${CMAKE_SOURCE_DIR}/kernels/kernel_addition.cpp" "task_addition" "250" "0.9" "xcvu9p-flga2104-2-e" "-I${CMAKE_SOURCE_DIR}/kernels")

add_custom_target(compile_swemu
        ${SDAccel_XOCC}
        -g 
        -c -t sw_emu
        --platform $ENV{AWS_PLATFORM} -R2
        #--profile_kernel data:all:all:all
        #--profile_kernel stall:all:all
        --profile_kernel exec:all:all
        --kernel task_addition
        --include ${CMAKE_SOURCE_DIR}/kernels/
        ${CMAKE_SOURCE_DIR}/kernels/kernel_addition.cpp
        -o kernel_addition_swemu.xo)

add_custom_target(compile_hwemu
        ${SDAccel_XOCC}
        -g 
        -c -t hw_emu
        --platform $ENV{AWS_PLATFORM} -R2
        #--profile_kernel data:all:all:all
        #--profile_kernel stall:all:all
        --profile_kernel exec:all:all
        --kernel task_addition
        --include ${CMAKE_SOURCE_DIR}/kernels/
        ${CMAKE_SOURCE_DIR}/kernels/kernel_addition.cpp
        -o kernel_addition_hwemu.xo)

add_custom_target(compile_hw
        ${SDAccel_XOCC}
        -g 
        -c -t hw
        --platform $ENV{AWS_PLATFORM} -R2
        #--profile_kernel data:all:all:all
        #--profile_kernel stall:all:all
        --profile_kernel exec:all:all
        --kernel task_addition
        --include ${CMAKE_SOURCE_DIR}/kernels/
        -R1
        ${CMAKE_SOURCE_DIR}/kernels/kernel_addition.cpp
        -o kernel_addition_hw.xo)

add_custom_target(link_swemu
        ${SDAccel_XOCC}
        -g 
        -l -t sw_emu
        --platform $ENV{AWS_PLATFORM}
        #--profile_kernel data:all:all:all
        #--profile_kernel stall:all:all
        --profile_kernel exec:all:all
        --optimize 0
        --nk task_addition:1
        --sp task_addition_1.inputTn1:bank1
        --sp task_addition_1.inputTn2:bank1
        --sp task_addition_1.outputTn:bank1
        kernel_addition_swemu.xo
        -o fpga_image.xclbin)

add_custom_target(link_hwemu
        ${SDAccel_XOCC}
        -g 
        -l -t hw_emu
        --platform $ENV{AWS_PLATFORM}
        #--profile_kernel data:all:all:all
        #--profile_kernel stall:all:all
        --profile_kernel exec:all:all
        --optimize 0
        --nk task_addition:1
        --sp task_addition_1.inputTn1:bank1
        --sp task_addition_1.inputTn2:bank1
        --sp task_addition_1.outputTn:bank1
        kernel_addition_hwemu.xo
        -o fpga_image.xclbin)

add_custom_target(link_hw
        ${SDAccel_XOCC}
        -g 
        -l -t hw
        --platform $ENV{AWS_PLATFORM}
        #--profile_kernel data:all:all:all
        #--profile_kernel stall:all:all
        --profile_kernel exec:all:all
        --optimize 0
        --nk task_addition:1
        --sp task_addition_1.inputTn1:bank1
        --sp task_addition_1.inputTn2:bank1
        --sp task_addition_1.outputTn:bank1
        kernel_addition_hw.xo
        -o fpga_image.xclbin)

SET(platform $ENV{AWS_PLATFORM})
configure_file(${CMAKE_SOURCE_DIR}/scripts/launch_swemu.sh ${CMAKE_BINARY_DIR}/launch_swemu.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/launch_hwemu.sh ${CMAKE_BINARY_DIR}/launch_hwemu.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/sdaccel.ini ${CMAKE_BINARY_DIR}/sdaccel.ini COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_swemu.sh ${CMAKE_BINARY_DIR}/autobuild_swemu.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_hwemu.sh ${CMAKE_BINARY_DIR}/autobuild_hwemu.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_hw.sh ${CMAKE_BINARY_DIR}/autobuild_hw.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/PasteBinAgentForAutoBuilds.py ${CMAKE_BINARY_DIR}/PasteBinAgentForAutoBuilds.py @ONLY)
