/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
