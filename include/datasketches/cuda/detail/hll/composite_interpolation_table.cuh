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

#include <cstdint>

#include <cuda_runtime.h>

namespace datasketches::cuda::detail::hll::composite_interpolation {

inline constexpr std::uint8_t min_lg_k       = 4;
inline constexpr std::uint8_t max_lg_k       = 18;
inline constexpr std::uint32_t num_x_values  = 257;
inline constexpr std::uint32_t last_x_offset = num_x_values - 1;

// DataSketches C++ 5.2.0, commit de8553ba372e618382c2e7b44b0ffc9422b9458c:
// hll/include/CompositeInterpolationXTable-internal.hpp.
//
// The source table contains lgK 4..21. CCCL currently supports lgK 4..18, so
// only those rows are checked in here. Keep host and device storage under the
// same name so host/device estimator code uses exactly the same values.
#if defined(__CUDA_ARCH__)
#define DATASKETCHES_CUDA_HLL_TABLE_DECL static __device__ constexpr
#else
#define DATASKETCHES_CUDA_HLL_TABLE_DECL inline constexpr
#endif

DATASKETCHES_CUDA_HLL_TABLE_DECL std::uint32_t y_strides[] = {
  1, 2, 3, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240};

DATASKETCHES_CUDA_HLL_TABLE_DECL double x_values[15][num_x_values] = {
#include <datasketches/cuda/detail/hll/composite_interpolation_table_data.inl>
};

#undef DATASKETCHES_CUDA_HLL_TABLE_DECL

[[nodiscard]] __host__ __device__ constexpr bool valid_lg_k(std::uint8_t lg_k) noexcept
{
  return lg_k >= min_lg_k && lg_k <= max_lg_k;
}

[[nodiscard]] __host__ __device__ inline const double* x_values_for(std::uint8_t lg_k) noexcept
{
  return x_values[lg_k - min_lg_k];
}

[[nodiscard]] __host__ __device__ inline std::uint32_t y_stride_for(std::uint8_t lg_k) noexcept
{
  return y_strides[lg_k - min_lg_k];
}

}  // namespace datasketches::cuda::detail::hll::composite_interpolation
