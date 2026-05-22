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

// normalizing_hasher<Key>: per-key normalization that mirrors
// datasketches-cpp's hll_sketch_alloc::update(...) overloads before hashing.
//
// datasketches-cpp routes all primitive types through MurmurHash3_x64_128
// operating on a canonical 8-byte representation (HllSketch-internal.hpp):
//
//   uint64_t / int64_t    → hash 8 raw bytes
//   int32/16/8_t          → sign-extend to int64_t, hash 8 bytes
//   uint32/16/8_t         → reinterpret as signed counterpart first, then
//                           sign-extend to int64_t, hash 8 bytes
//   double                → canonicalize -0.0 → +0.0,
//                           NaN → 0x7ff8000000000000 (Java Double.doubleToLongBits),
//                           hash 8 bytes
//   float                 → convert to double, then apply double canonicalization
//
// Unsupported types (strings, spans) produce a compile error via static_assert.

#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cuda_runtime.h>

#include <cuda/std/__bit/bit_cast.h>

#include <cuda/experimental/__cuco/hash_functions.cuh>

namespace datasketches::cuda::detail {

// Inner hasher: MurmurHash3_x64_128 operating on uint64_t (the canonical
// 8-byte representation after normalization).
using _murmur_u64 =
  ::cuda::experimental::cuco::hash<::cuda::std::uint64_t,
                                   ::cuda::experimental::cuco::hash_algorithm::murmurhash3_x64_128>;

template <class _Key>
class normalizing_hasher {
 public:
  using result_type = __uint128_t;

  __host__ __device__ constexpr normalizing_hasher(::cuda::std::uint64_t __seed = 9001) noexcept
    : _inner(__seed)
  {
  }

  [[nodiscard]] __host__ __device__ constexpr __uint128_t operator()(const _Key& __k) const noexcept
  {
    return _inner(_canonicalize(__k));
  }

 private:
  _murmur_u64 _inner;

  // Produce the canonical uint64_t that datasketches-cpp would hash for __k.
  __host__ __device__ static constexpr ::cuda::std::uint64_t _canonicalize(const _Key& __k) noexcept
  {
    // uint64_t / int64_t: hash raw bits (no extension needed).
    if constexpr (sizeof(_Key) == 8 && ::cuda::std::is_integral_v<_Key>) {
      return ::cuda::std::bit_cast<::cuda::std::uint64_t>(__k);
    }
    // Narrow signed integrals: sign-extend to int64_t.
    // Matches HllSketch-internal.hpp update(int32/16/8_t):
    //   const int64_t val = static_cast<int64_t>(datum);
    else if constexpr (::cuda::std::is_integral_v<_Key> && ::cuda::std::is_signed_v<_Key> &&
                       sizeof(_Key) < 8) {
      return ::cuda::std::bit_cast<::cuda::std::uint64_t>(static_cast<::cuda::std::int64_t>(__k));
    }
    // Narrow unsigned integrals: reinterpret as signed counterpart first, then
    // sign-extend. Mirrors update(uint32_t) → update(int32_t) → int64_t, etc.
    else if constexpr (::cuda::std::is_integral_v<_Key> && !::cuda::std::is_signed_v<_Key> &&
                       sizeof(_Key) < 8) {
      using _Signed = ::cuda::std::make_signed_t<_Key>;
      return ::cuda::std::bit_cast<::cuda::std::uint64_t>(
        static_cast<::cuda::std::int64_t>(static_cast<_Signed>(__k)));
    }
    // float: convert to double, then apply the same canonicalization as double.
    // Inline rather than calling _canonicalize(double) because _canonicalize is
    // a member template on _Key — calling it with a double argument while
    // _Key=float would bind to `const float&` via implicit narrowing conversion,
    // reconstructing a float and looping forever.
    else if constexpr (::cuda::std::is_same_v<_Key, float>) {
      double __d = static_cast<double>(__k);
      if (::cuda::std::isnan(__d)) { return ::cuda::std::uint64_t{0x7ff8000000000000ULL}; }
      if (__d == 0.0) __d = 0.0;  // -0.0 → +0.0
      return ::cuda::std::bit_cast<::cuda::std::uint64_t>(__d);
    }
    // double: canonicalize -0.0 → +0.0 and NaN → Java quiet-NaN canonical.
    // Mirrors update(double) in HllSketch-internal.hpp:179-190.
    else if constexpr (::cuda::std::is_same_v<_Key, double>) {
      if (::cuda::std::isnan(__k)) { return ::cuda::std::uint64_t{0x7ff8000000000000ULL}; }
      double __d = (__k == 0.0) ? 0.0 : __k;  // -0.0 → +0.0
      return ::cuda::std::bit_cast<::cuda::std::uint64_t>(__d);
    } else {
      // Type-dependent false to prevent eager instantiation in non-C++20 mode.
      static_assert(sizeof(_Key) == 0,
                    "datasketches::cuda::normalizing_hasher: unsupported Key type. "
                    "Supported: int8/16/32/64_t, uint8/16/32/64_t, float, double. "
                    "String / byte-range support requires a separate span-based API.");
    }
  }
};

}  // namespace datasketches::cuda::detail
