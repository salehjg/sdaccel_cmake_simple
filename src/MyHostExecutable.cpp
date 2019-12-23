//
// Created by saleh on 11/11/19.
//
#include "xcl2.hpp"
#include <vector>

#define BATCH 2
#define SLICE 4
#define LENGTH (BATCH*SLICE)

int main(int argc, char **argv) {
    if(argc!=2){
        std::cout <<"Please enter the abs path for xclbin or awsxclbin file as the first and only argument.\nAborting...\n";
    }
    std::string BinaryFile(argv[1]);
    //------------------------------------------------------------------------------------------
    cl_int err;
    std::vector<float, aligned_allocator<float>> h_a(LENGTH); //host memory for a vector
    std::vector<float, aligned_allocator<float>> h_b(LENGTH); //host memory for b vector
    std::vector<float, aligned_allocator<float>> h_c(LENGTH); //host memory for c vector
    std::vector<float, aligned_allocator<float>> h_result(LENGTH); //host memory for c vector

    //Fill our data sets with pattern
    int i = 0;
    for (i = 0; i < LENGTH; i++) {
        h_a[i] = i;
        h_b[i] = i;
        h_c[i] = h_a[i] + h_b[i];
        h_result[i] = 0;
    }

    auto devices = xcl::get_xil_devices();
    auto device = devices[0];

    //Creating Context and Command Queue for selected Device
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err,cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    int batchsize = BATCH;
    int slice_len = SLICE;
    bool match = true;


    {
        printf("INFO: loading kernel\n");
        //std::string BinaryFile = "fpga_image.xclbin"; (now the host program gets this as an argument)
        auto fileBuf = xcl::read_binary_file(BinaryFile);
        cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
        devices.resize(1);
        OCL_CHECK(err,cl::Program program(context, devices, bins, NULL, &err));
        OCL_CHECK(err,cl::Kernel krnl_addition(program, "task_addition", &err));

        OCL_CHECK(err,
                  cl::Buffer d_a(context,
                                 CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                 sizeof(float) * LENGTH,
                                 h_a.data(),
                                 &err));
        OCL_CHECK(err,
                  cl::Buffer d_b(context,
                                 CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                 sizeof(float) * LENGTH,
                                 h_b.data(),
                                 &err));
        OCL_CHECK(err,
                  cl::Buffer d_result(context,
                                   CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(float) * LENGTH,
                                   h_result.data(),
                                   &err));

        OCL_CHECK(err, err = krnl_addition.setArg(0, d_a));
        OCL_CHECK(err, err = krnl_addition.setArg(1, d_b));
        OCL_CHECK(err, err = krnl_addition.setArg(2, d_result));
        OCL_CHECK(err, err = krnl_addition.setArg(3, batchsize));
        OCL_CHECK(err, err = krnl_addition.setArg(4, slice_len));

        OCL_CHECK(err,err = q.enqueueMigrateMemObjects({d_a, d_b},0 /* 0 means from host*/));

        // This function will execute the kernel on the FPGA
        OCL_CHECK(err,err = q.enqueueTask(krnl_addition));
        OCL_CHECK(err,err = q.enqueueMigrateMemObjects({d_result},CL_MIGRATE_MEM_OBJECT_HOST));
        OCL_CHECK(err,err = q.finish());

        // Check Results
        for (int i = 0; i < LENGTH; i++) {
            if ((h_c[i]) != h_result[i]) {
                printf("ERROR - %d - a=%f, b=%f, c=%f, c_fpga=%f\n",
                       i,
                       h_a[i],
                       h_b[i],
                       h_c[i],
                       h_result[i]);
                match = false;
                //break;
            }
        }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
