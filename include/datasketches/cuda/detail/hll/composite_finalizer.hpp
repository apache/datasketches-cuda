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

#include <datasketches/cuda/detail/hll/composite_finalizer.cuh>

namespace datasketches::cuda::detail::hll {

//! @brief DataSketches Composite estimate from the serialized HLL reduction state.
//!
//! @param[in] z Sum of `2^-register[i]` across all registers.
//! @param[in] cur_min Minimum register value.
//! @param[in] num_at_cur_min Number of registers equal to `cur_min`.
//! @param[in] lg_k HLL precision parameter.
//! @return The Composite cardinality estimate.
[[nodiscard]] inline double composite_finalizer(double z,
                                                std::uint8_t cur_min,
                                                std::uint32_t num_at_cur_min,
                                                std::uint8_t lg_k) noexcept
{
  const std::uint32_t num_zeroes = cur_min == 0 ? num_at_cur_min : 0;
  return composite_estimate(z, num_zeroes, lg_k);
}

}  // namespace datasketches::cuda::detail::hll
