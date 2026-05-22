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

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cuda_runtime.h>

#include <cuda/std/__bit/countl.h>

#include <cuda/experimental/__cuco/hash_functions.cuh>

#include <datasketches/cuda/detail/hll/normalizing_hasher.cuh>

namespace datasketches::cuda::detail {

//! @brief Policy that drives `cuda::experimental::cuco::hyperloglog` to produce
//! sketches binary-compatible with `datasketches::hll_sketch` (apache datasketches-cpp).
//!
//! Hash: MurmurHash3_x64_128 with seed 9001 (datasketches-cpp `DEFAULT_SEED`,
//! `common_defs.hpp:34`). The 128-bit hash is split into h1/h2 matching
//! `datasketches::HllUtil::coupon` (`HllUtil.hpp:141-146`). The low 64 bits of
//! the cudax `__uint128_t` correspond to datasketches `hashState.h1` and the
//! high 64 to `h2`, given the cudax internal layout
//! `bit_cast<__uint128_t>(uint64_t[2]{h0, h1})`.
//!
//! Bit-slicing:
//!   register_index = h1 & ((1 << precision) - 1)        // low `precision` bits of h1
//!   register_value = min(countl_zero(h2), 62) + 1       // rho, capped at 63
//!
//! The slot derivation matches `Hll8Array::couponUpdate` (`Hll8Array-internal.hpp:96`):
//! `slotNo = HllUtil::getLow26(coupon) & configKmask = h1 & ((1<<lgK)-1)` since
//! lgK <= 21 < 26. The 62-cap matches `HllUtil::coupon` line 144.
//!
//! `finalize` is a no-op stub satisfying the cudax policy concept. The downstream
//! `hll_sketch::get_estimate` bypasses cudax's `estimate()` path entirely (it
//! D2H-copies the registers, runs a wider reduction, and applies the Composite
//! estimator on host).
template <class _Key>
struct datasketches_policy {
  // Per-key normalization matching datasketches-cpp's hll_sketch::update(...)
  // overloads (HllSketch-internal.hpp). normalizing_hasher<_Key> converts the
  // key to its canonical 8-byte representation before feeding
  // MurmurHash3_x64_128, so GPU coupons are byte-compatible with the CPU sketch
  // for all supported types. Unsupported types produce a compile error in
  // normalizing_hasher::_canonicalize.
  using hasher           = normalizing_hasher<_Key>;
  using hash_result_type = __uint128_t;
  using register_type    = ::cuda::std::int32_t;

  //! @brief Default datasketches HLL hash seed (`common_defs.hpp:34`).
  static constexpr ::cuda::std::uint64_t default_seed = 9001;

  hasher hasher_{default_seed};

  //! @brief Returns the underlying hash functor.
  //!
  //! @return The hash functor.
  [[nodiscard]] __host__ __device__ constexpr hasher hash_function() const noexcept { return hasher_; }

  //! @brief Hashes an item.
  //!
  //! @param[in] __k The item to hash.
  //! @return The 128-bit MurmurHash3 hash value of `__k`.
  [[nodiscard]] __host__ __device__ constexpr hash_result_type hash(const _Key& __k) const noexcept
  {
    return hasher_(__k);
  }

  //! @brief Extracts the register index from the hash.
  //!
  //! @note Matches `Hll8Array::couponUpdate` which AND-masks the low 26 bits of h1
  //! with `((1 << lgK) - 1)`. For lgK <= 21 this reduces to `h1 & ((1 << lgK) - 1)`.
  //!
  //! @param[in] __h The 128-bit hash value.
  //! @param[in] __precision The HLL precision parameter (lgK).
  //! @return The register index in `[0, 2^__precision)`.
  [[nodiscard]] __host__ __device__ constexpr ::cuda::std::uint32_t register_index(
    hash_result_type __h, int __precision) const noexcept
  {
    const auto __h1   = static_cast<::cuda::std::uint64_t>(__h);
    const auto __mask = (::cuda::std::uint64_t{1} << __precision) - 1;
    return static_cast<::cuda::std::uint32_t>(__h1 & __mask);
  }

  //! @brief Computes rho (1 + leading zeros of h2, with the leading-zero count
  //! capped at 62 to bound rho to the range [1, 63]).
  //!
  //! @note Matches `HllUtil::coupon` (`HllUtil.hpp:143-144`):
  //! `lz = clz(h2); value = min(lz, 62) + 1`.
  //!
  //! @param[in] __h The 128-bit hash value.
  //! @param[in] __precision Unused; present for cudax policy concept conformance.
  //! @return rho, in `[1, 63]`.
  [[nodiscard]] __host__ __device__ constexpr ::cuda::std::uint8_t register_value(
    hash_result_type __h, int __precision) const noexcept
  {
    (void)__precision;
    const auto __h2  = static_cast<::cuda::std::uint64_t>(__h >> 64);
    const auto __lz  = static_cast<::cuda::std::uint8_t>(::cuda::std::countl_zero(__h2));
    const auto __cap = static_cast<::cuda::std::uint8_t>(__lz > 62 ? 62 : __lz);
    return static_cast<::cuda::std::uint8_t>(__cap + 1);
  }

  //! @brief No-op finalizer satisfying the cudax policy concept.
  //!
  //! @note Never invoked by `datasketches::cuda::hll_sketch`. Compute the
  //! estimate via the Composite estimator on host-side after a D2H copy of the
  //! register array. See `composite_finalizer.hpp`.
  //!
  //! @return 0 unconditionally.
  [[nodiscard]] __host__ __device__ static constexpr ::cuda::std::size_t finalize(double, int, int) noexcept
  {
    return 0;
  }
};

}  // namespace datasketches::cuda::detail
