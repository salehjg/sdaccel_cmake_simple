//
// Created by saleh on 11/11/19.
//

#define MAX_SLICE_LEN   4

template <typename DType>
void addition(
    DType *inputTn1, //Same as MatA
    DType *inputTn2, //Same as MatB
    DType *outputTn,
    unsigned int dim0, //Batch
    unsigned int dim1,  //len
    unsigned int repeat ){
    float buff1[MAX_SLICE_LEN];
    float buff2[MAX_SLICE_LEN];
    float buff3[MAX_SLICE_LEN];

    LoopBatch:for(int batch=0; batch<dim0; batch++){
    #pragma HLS LOOP_TRIPCOUNT min=5 max=5

        //1. Loading the current slices into the local buffers
        LoopLoad:for(int d1=0; d1<MAX_SLICE_LEN; d1++){
            if(d1<dim1){
                buff1[d1] = inputTn1[batch*dim1 + d1];
                buff2[d1] = inputTn2[batch*dim1 + d1];
                buff3[d1] = 0; //wiping the result buffer
            }else{
                buff1[d1] = 0;
                buff2[d1] = 0;
            }
        }

        //2. Processing
        LoopRepeat:for(int p=0; p<repeat; p++) {
            LoopProcess:for (int d1 = 0; d1 < MAX_SLICE_LEN; d1++) {
                buff3[d1] = buff1[d1] + buff2[d1];
            }
        }

        //3. Storing the result in global memory(output tensor)
        LoopStore:for(int d1=0; d1<MAX_SLICE_LEN; d1++){
            if(d1<dim1){
                outputTn[batch*dim1 + d1] = buff3[d1];
            }
        }

    }
}

extern "C"{
void task_addition(
        float *inputTn1,
        float *inputTn2,
        float *outputTn,
        unsigned int dim0,
        unsigned int dim1){
#pragma HLS INTERFACE m_axi     port=inputTn1   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=inputTn2   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn1   bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0A      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1A      bundle=control

#pragma HLS INTERFACE s_axilite port=dim0B      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1B      bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

    addition<float>(inputTn1, inputTn2, outputTn, dim0, dim1, 2);
}
}
