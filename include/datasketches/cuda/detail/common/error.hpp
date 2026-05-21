#pragma once

#include <string>

#include <cuda_runtime.h>

#include <datasketches/cuda/common/error.hpp>

#define DATASKETCHES_CUDA_STRINGIFY_DETAIL(x) #x
#define DATASKETCHES_CUDA_STRINGIFY(x)        DATASKETCHES_CUDA_STRINGIFY_DETAIL(x)

//! @brief Error-checking wrapper for CUDA runtime API calls.
//!
//! Invokes `_call`; on any non-`cudaSuccess` status, clears the sticky error
//! via `cudaGetLastError()` and throws the exception type with a message
//! containing the file/line, error name, and error string.
//!
//! Usage:
//! @code
//!   DATASKETCHES_CUDA_TRY(cudaMallocAsync(&p, n, stream));
//!   DATASKETCHES_CUDA_TRY(cudaMallocAsync(&p, n, stream), std::runtime_error);
//! @endcode
#define DATASKETCHES_CUDA_TRY(...)                                                               \
  GET_DATASKETCHES_CUDA_TRY_MACRO(__VA_ARGS__, DATASKETCHES_CUDA_TRY_2, DATASKETCHES_CUDA_TRY_1) \
  (__VA_ARGS__)
#define GET_DATASKETCHES_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME
#define DATASKETCHES_CUDA_TRY_2(_call, _exception_type)                                         \
  do {                                                                                          \
    cudaError_t const status__ = (_call);                                                       \
    if (cudaSuccess != status__) {                                                              \
      cudaGetLastError();                                                                       \
      throw _exception_type(                                                                    \
        std::string{"CUDA error at " __FILE__ ":" DATASKETCHES_CUDA_STRINGIFY(__LINE__) ": "} + \
        cudaGetErrorName(status__) + " " + cudaGetErrorString(status__));                       \
    }                                                                                           \
  } while (0)
#define DATASKETCHES_CUDA_TRY_1(_call) \
  DATASKETCHES_CUDA_TRY_2(_call, ::datasketches::cuda::cuda_error)
