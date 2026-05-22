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
#include <cuda/std/span>

#include <inv_pow2_table.hpp>

namespace datasketches::cuda::detail {

//! @brief State produced by the wider reduction over an HLL_8 register array.
//!
//! Mirrors the four state fields tracked by `datasketches::HllArray` that the
//! Composite estimator reads: `kxq0`, `kxq1`, `curMin`, `numAtCurMin`. The HIP
//! accumulator is intentionally absent: the GPU forces `oooFlag=true` on
//! serialize, which pins the CPU side to Composite (HIP requires a serial
//! observation order that the parallel atomic-max kernel does not provide).
struct reduction_result {
  //! Sum of `2^{-r_i} - 1.0` over registers with `0 < r_i < 32`, plus an
  //! initial `2^{lgK}` to account for `r_i == 0` registers contributing 1.0.
  double kxq0;
  //! Same sum but over registers with `r_i >= 32`.
  double kxq1;
  //! Minimum register value across the array. For HLL_8 with at least one
  //! zero register, this is 0; otherwise the minimum non-zero rho.
  std::uint8_t cur_min;
  //! Count of registers equal to `cur_min`.
  std::uint32_t num_at_cur_min;
};

//! @brief Reduce an HLL_8 register array to the Composite-estimator state.
//!
//! Operates on the cudax HLL register storage, which is `int32_t` per slot
//! holding rho directly in `[0, 63]`.
//!
//! For HLL_8, the CPU sketch maintains the invariant `curMin == 0` and
//! `numAtCurMin == #{i : reg[i] == 0}` regardless of whether the array still
//! contains zero registers. This matches `Hll8Array::couponUpdate`'s incremental
//! accounting (`Hll8Array-internal.hpp:88-104`), where transitioning a register
//! from 0 to non-zero decrements `numAtCurMin_` and `curMin_` is never raised.
//! The Composite estimator's bit-map branch (`HllArray-internal.hpp:563-574`)
//! falls back to `configK * log(configK/0.5)` when `numUnhitBuckets == 0`, so
//! holding `curMin == 0` even on a fully saturated array yields the same answer
//! as a real-minimum scheme; the difference is only on the wire (byte 6 is
//! always 0 for HLL_8).
//!
//! `kxq0`/`kxq1` follow `HllArray::check_rebuild_kxq_cur_min`
//! (`HllArray-internal.hpp:600-625`): split-precision sum of `2^{-r_i}` over
//! all registers, with `r_i < 32` going to `kxq0` and `r_i >= 32` to `kxq1`.
//! Both accumulators start with one unit per register (configK / 0) and are
//! adjusted by `INVERSE_POWERS_OF_2[r] - 1.0` for each non-zero register.
//!
//! @param[in] registers The host-side copy of the cudax HLL register array.
//! @param[in] lgK The HLL precision parameter; `registers.size()` must equal
//!   `1 << lgK`.
//! @return The reduction state ready to feed into `composite_finalizer`.
inline reduction_result reduce_hll8(::cuda::std::span<const std::int32_t> registers,
                                    std::uint8_t lgK) noexcept
{
  const std::uint32_t configK = 1u << lgK;

  reduction_result r{};
  r.kxq0           = static_cast<double>(configK);
  r.kxq1           = 0.0;
  r.cur_min        = 0;
  r.num_at_cur_min = 0;

  for (std::uint32_t i = 0; i < configK; ++i) {
    const auto v = static_cast<std::uint8_t>(registers[i]);

    if (v == 0) {
      ++r.num_at_cur_min;
      continue;
    }

    const double delta = ::datasketches::INVERSE_POWERS_OF_2[v] - 1.0;
    if (v < 32) {
      r.kxq0 += delta;
    } else {
      r.kxq1 += delta;
    }
  }

  return r;
}

}  // namespace datasketches::cuda::detail
