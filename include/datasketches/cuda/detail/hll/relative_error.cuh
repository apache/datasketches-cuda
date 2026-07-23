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
#include <cuda/std/cmath>

#include <cuda_runtime.h>

namespace datasketches::cuda::detail::hll {

// DataSketches C++ 5.2.0, commit de8553ba372e618382c2e7b44b0ffc9422b9458c:
// hll/include/RelativeErrorTables-internal.hpp.
#if defined(__CUDA_ARCH__)
#define DATASKETCHES_CUDA_HLL_REL_ERROR_DECL static __device__ constexpr
#else
#define DATASKETCHES_CUDA_HLL_REL_ERROR_DECL inline constexpr
#endif

DATASKETCHES_CUDA_HLL_REL_ERROR_DECL double non_hip_lower_bound_relative_error[] = {
  0.254409839, 0.682266712, 1.304022158,  // lgK 4
  0.181817353, 0.443389054, 0.778776219,  // lgK 5
  0.129432281, 0.295782195, 0.49252279,   // lgK 6
  0.091640655, 0.201175925, 0.323664385,  // lgK 7
  0.064858051, 0.138523393, 0.218805328,  // lgK 8
  0.045851855, 0.095925072, 0.148635751,  // lgK 9
  0.032454144, 0.067009668, 0.102660669,  // lgK 10
  0.022921382, 0.046868565, 0.071307398,  // lgK 11
  0.016155679, 0.032825719, 0.049677541   // lgK 12
};

DATASKETCHES_CUDA_HLL_REL_ERROR_DECL double non_hip_upper_bound_relative_error[] = {
  -0.256980172, -0.411905944, -0.52651057,   // lgK 4
  -0.182332109, -0.310275547, -0.412660505,  // lgK 5
  -0.129314228, -0.230142294, -0.315636197,  // lgK 6
  -0.091584836, -0.16834013,  -0.236346847,  // lgK 7
  -0.06487411,  -0.122045231, -0.174112107,  // lgK 8
  -0.04591465,  -0.08784505,  -0.126917615,  // lgK 9
  -0.032433119, -0.062897613, -0.091862929,  // lgK 10
  -0.022960633, -0.044875401, -0.065736049,  // lgK 11
  -0.016186662, -0.031827816, -0.046973459   // lgK 12
};

#undef DATASKETCHES_CUDA_HLL_REL_ERROR_DECL

//! @brief Return the DataSketches non-HIP relative error for a confidence bound.
//!
//! @pre `lg_k` is in `[4, 18]`.
//! @pre `num_std_dev` is in `[1, 3]`.
[[nodiscard]] __host__ __device__ inline double relative_error(bool upper_bound,
                                                               std::uint8_t lg_k,
                                                               std::uint8_t num_std_dev) noexcept
{
  if (lg_k > 12) {
    constexpr double non_hip_rse_factor = 1.03896;
    const auto config_k                 = static_cast<double>(std::uint32_t{1} << lg_k);
    const double error = (num_std_dev * non_hip_rse_factor) / ::cuda::std::sqrt(config_k);
    return upper_bound ? -error : error;
  }

  const auto index = static_cast<std::uint32_t>((lg_k - 4) * 3 + (num_std_dev - 1));
  return upper_bound ? non_hip_upper_bound_relative_error[index]
                     : non_hip_lower_bound_relative_error[index];
}

}  // namespace datasketches::cuda::detail::hll
