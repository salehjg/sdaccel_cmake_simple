cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)

PROJECT(SDAccel_CMake_Simple)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)
find_package(SDAccel REQUIRED)

add_executable(MyHostExecutable src/MyHostExecutable.cpp src/xcl2.cpp)
include_directories(${SDAccel_INCLUDE_DIRS} inc/)
target_link_libraries(MyHostExecutable ${SDAccel_LIBRARIES})



add_custom_target(compile_swemu
        ${SDAccel_XOCC}
        -c -t sw_emu
        --platform $ENV{AWS_PLATFORM} -R2
        --profile_kernel data:all:all:all
        --profile_kernel stall:all:all
        --profile_kernel exec:all:all
        --kernel task_addition
        ${CMAKE_SOURCE_DIR}/kernels/kernel_addition.cpp
        -o kernel_addition_swemu.xo)

add_custom_target(compile_hwemu
        ${SDAccel_XOCC}
        -c -t hw_emu
        --platform $ENV{AWS_PLATFORM} -R2
        --profile_kernel data:all:all:all
        --profile_kernel stall:all:all
        --profile_kernel exec:all:all
        --kernel task_addition
        ${CMAKE_SOURCE_DIR}/kernels/kernel_addition.cpp
        -o kernel_addition_hwemu.xo)

add_custom_target(compile_hw
        ${SDAccel_XOCC}
        -c -t hw
        --platform $ENV{AWS_PLATFORM} -R2
        --profile_kernel data:all:all:all
        --profile_kernel stall:all:all
        --profile_kernel exec:all:all
        --kernel task_addition
        ${CMAKE_SOURCE_DIR}/kernels/kernel_addition.cpp
        -o kernel_addition_hw.xo)

add_custom_target(link_swemu
        ${SDAccel_XOCC}
        -l -t sw_emu
        --platform $ENV{AWS_PLATFORM}
        --optimize 0
        --nk task_addition:1
        --sp task_addition_1.inputTn1:bank1
        --sp task_addition_1.inputTn2:bank1
        --sp task_addition_1.outputTn:bank1
        kernel_addition_swemu.xo
        -o fpga_image.xclbin)

add_custom_target(link_hwemu
        ${SDAccel_XOCC}
        -l -t hw_emu
        --platform $ENV{AWS_PLATFORM}
        --optimize 0
        --nk task_addition:1
        --sp task_addition_1.inputTn1:bank1
        --sp task_addition_1.inputTn2:bank1
        --sp task_addition_1.outputTn:bank1
        kernel_addition_hwemu.xo
        -o fpga_image.xclbin)

add_custom_target(link_hw
        ${SDAccel_XOCC}
        -l -t hw
        --platform $ENV{AWS_PLATFORM}
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
