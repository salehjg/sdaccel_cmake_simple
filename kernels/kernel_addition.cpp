//
// Created by saleh on 11/11/19.
//
#include <stdio.h>
#include "VectorizationHelper.h"
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include <ap_fixed.h>
#include <cassert>

// 16 words of 4-bytes floats
constexpr int __vecDepth = 16;
constexpr int __maxSliceLen = 16;

using Data_t = float;
using MemoryPackK_t = hlslib::DataPack<Data_t, __vecDepth>;
using hlslib::Stream;

void AdditionBad(
    MemoryPackK_t* inputTn1,
    MemoryPackK_t* inputTn2,
    MemoryPackK_t* outputTn,
    unsigned int dim0, //Batch
    unsigned int dim1  //len
    ){

    float buff1[__maxSliceLen];
    float buff2[__maxSliceLen];
    float buff3[__maxSliceLen];

    LoopBatch:for(int batch=0; batch<dim0; batch++){
#pragma HLS LOOP_TRIPCOUNT min=2 max=2

        //1. Loading the current slices into the local buffers
        LoopLoad:for(int d1=0; d1<__maxSliceLen; d1++){
            if(d1<dim1){
                buff1[d1] = inputTn1[
                        FlatIdx_to_VecIdx(__vecDepth, batch*dim1 + d1)
                        ][FlatIdx_to_VecSubIdx(__vecDepth,batch*dim1 + d1)];
                buff2[d1] = inputTn2[
                        FlatIdx_to_VecIdx(__vecDepth, batch*dim1 + d1)
                        ][FlatIdx_to_VecSubIdx(__vecDepth,batch*dim1 + d1)];
                buff3[d1] = 0; //wiping the result buffer
            }else{
                buff1[d1] = 0;
                buff2[d1] = 0;
            }
        }

        //2. Processing
        LoopProcess:for (int d1 = 0; d1 < __maxSliceLen; d1++) {
            buff3[d1] = buff1[d1] + buff2[d1];
        }


        //3. Storing the result in global memory(output tensor)
        LoopStore:for(int d1=0; d1<__maxSliceLen; d1++){
            if(d1<dim1){
                outputTn[FlatIdx_to_VecIdx(__vecDepth,batch*dim1 + d1)][
                        FlatIdx_to_VecSubIdx(__vecDepth,batch*dim1 + d1)] = buff3[d1];
            }
        }

    }
}

void AdditionGoodRead(
    MemoryPackK_t* inputTn,
    Stream<MemoryPackK_t, 1>& streamOut,
    int offset,
    int len){

    LoopRead: for(int i=0; i<len; i++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
        streamOut.Push(inputTn[offset + i]);
    }

}

void AdditionGoodProcessingEelement(
    Stream<MemoryPackK_t, 1>& streamIn1,
    Stream<MemoryPackK_t, 1>& streamIn2,
    Stream<MemoryPackK_t, 1>& streamOut,
    int len){

    LoopRead: for(int i=0; i<len; i++){
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
#pragma HLS PIPELINE II=1

        MemoryPackK_t vec1, vec2, vecOut;
        vec1 = streamIn1.Pop();
        vec2 = streamIn2.Pop();

        LoopCUs: for(int c=0; c<__vecDepth; c++){
#pragma HLS UNROLL
            vecOut[c] = vec1[c] + vec2[c];
        }

        streamOut.Push(vecOut);
    }

}

void AdditionGoodWrite(
    Stream<MemoryPackK_t, 1>& streamIn,
    MemoryPackK_t* outputTn,
    int offset,
    int len){
    
    LoopWrite: for(int i=0; i<len; i++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
        outputTn[offset + i] = streamIn.Pop();
    }

}

void AdditionGood(
    MemoryPackK_t* inputTn1,
    MemoryPackK_t* inputTn2,
    MemoryPackK_t* outputTn,
    unsigned int dim0,  //Batch
    unsigned int dim1   //len
    ){

    assert(__vecDepth==__maxSliceLen);
    assert(dim1==__vecDepth);


#pragma HLS DATAFLOW
    HLSLIB_DATAFLOW_INIT();
    Stream<MemoryPackK_t, 1> stream1, stream2, stream3;

    HLSLIB_DATAFLOW_FUNCTION(AdditionGoodRead, inputTn1, stream1, 0, dim0);
    HLSLIB_DATAFLOW_FUNCTION(AdditionGoodRead, inputTn2, stream2, 0, dim0);
    HLSLIB_DATAFLOW_FUNCTION(AdditionGoodProcessingEelement, stream1, stream2, stream3, dim0);
    HLSLIB_DATAFLOW_FUNCTION(AdditionGoodWrite, stream3, outputTn, 0, dim0);

    HLSLIB_DATAFLOW_FINALIZE();

}

extern "C"{
void task_addition(
        MemoryPackK_t *inputTn1,
        MemoryPackK_t *inputTn2,
        MemoryPackK_t *outputTn1,
        MemoryPackK_t *outputTn2,
        unsigned int dim0,
        unsigned int dim1){
#pragma HLS INTERFACE m_axi     port=inputTn1   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=inputTn2   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn1  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi     port=outputTn2  offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=inputTn1   bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn1  bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn2  bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1       bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

//#pragma HLS data_pack variable=inputTn1
//#pragma HLS data_pack variable=inputTn2
//#pragma HLS data_pack variable=outputTn

    AdditionGood(inputTn1, inputTn2, outputTn1, dim0, dim1);
    AdditionBad(inputTn1, inputTn2, outputTn2, dim0, dim1);
}
}
