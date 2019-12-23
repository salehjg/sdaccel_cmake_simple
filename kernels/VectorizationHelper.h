//
// Created by saleh on 12/23/19.
//

#ifndef SDACCEL_CMAKE_SIMPLE_VECTORIZATIONHELPER_H
#define SDACCEL_CMAKE_SIMPLE_VECTORIZATIONHELPER_H

template <typename DType, int VectorizationDepth>
struct VectorizedArray{
    DType vec[VectorizationDepth];
};

// Be careful with parenthesis in a preprocessor, especially when the expression has math ops!
#define FlatIdx_to_VecIdx(vecDepth, flatIdx) ((flatIdx)/(vecDepth))
#define FlatIdx_to_VecSubIdx(vecDepth, flatIdx) ((flatIdx)%(vecDepth))

#endif //SDACCEL_CMAKE_SIMPLE_VECTORIZATIONHELPER_H
