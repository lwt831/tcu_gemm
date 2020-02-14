#ifndef CHECKERROR
#define CHECKERROR
#include <assert.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <cstdio>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cstring> // memset

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure\nError: " << cudaGetErrorString(status); \
      std::stringstream _where, _message;                                \
      _where << __FILE__ << ':' << __LINE__;                             \
      _message << _error.str() + "\n" << __FILE__ << ':' << __LINE__;\
      std::cerr << _message.str() << "\nAborting...\n";                  \
      cudaDeviceReset();                                                 \
      exit(EXIT_FAILURE);                                                \
    }                                                                  \
}

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

#define checkCublasErrors(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf(stderr,"CUBLASassert Failure Code: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}

static const char * cusparseGetErrorString(cusparseStatus_t error)
{
    // Read more at: http://docs.nvidia.com/cuda/cusparse/index.html#ixzz3f79JxRar
    switch (error)
    {
    case CUSPARSE_STATUS_SUCCESS:
        return "The operation completed successfully.";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "The cuSPARSE library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSPARSE routine, or an error in the hardware setup.\n" \
               "To correct: call cusparseCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";
 
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "Resource allocation failed inside the cuSPARSE library. This is usually caused by a cudaMalloc() failure.\n"\
                "To correct: prior to the function call, deallocate previously allocated memory as much as possible.";
 
    case CUSPARSE_STATUS_INVALID_VALUE:
        return "An unsupported value or parameter was passed to the function (a negative vector size, for example).\n"\
            "To correct: ensure that all the parameters being passed have valid values.";
 
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision.\n"\
            "To correct: compile and run the application on a device with appropriate compute capability, which is 1.1 for 32-bit atomic operations and 1.3 for double precision.";
 
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\n"\
            "To correct: prior to the function call, unbind any previously bound textures.";
 
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\n"\
                "To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";
 
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "An internal cuSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\n"\
                "To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.";
 
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\n"\
                "To correct: check that the fields in cusparseMatDescr_t descrA were set correctly.";
    }
 
    return "<unknown>";
}
static void CudaSparseCheckCore(cusparseStatus_t code, const char *file, int line) {
   if (code != CUSPARSE_STATUS_SUCCESS) {
      fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, cusparseGetErrorString(code), file, line);
      exit(code);
   }
}
 
#define CudaSparseCheck( test ) { CudaSparseCheckCore((test), __FILE__, __LINE__); }


 #endif
