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

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <datasketches/cuda/detail/hll/normalizing_hasher.cuh>

namespace datasketches::cuda::detail::theta {

inline constexpr std::uint64_t default_seed = detail::hll::default_seed;
inline constexpr std::uint64_t max_theta    = 0x7fffffffffffffffULL;
inline constexpr std::uint8_t min_lg_k      = 5;
inline constexpr std::uint8_t max_lg_k      = 26;
inline constexpr std::uint8_t default_lg_k  = 12;

template <class Key>
struct theta_hash {
  detail::hll::normalizing_hasher<Key> hasher;

  __host__ __device__ explicit constexpr theta_hash(std::uint64_t seed) noexcept : hasher(seed) {}

  [[nodiscard]] __host__ __device__ constexpr std::uint64_t operator()(
    const Key& key) const noexcept
  {
    // DataSketches Theta uses the low MurmurHash3 word and an unsigned shift
    // by one, reserving zero as an empty-table sentinel.
    return static_cast<std::uint64_t>(hasher(key)) >> 1;
  }
};

struct screen_hash {
  std::uint64_t theta;

  [[nodiscard]] __host__ __device__ constexpr bool operator()(std::uint64_t hash) const noexcept
  {
    return hash != 0 && hash < theta;
  }
};

struct membership_filter {
  const std::uint64_t* other;
  std::size_t other_size;
  std::uint64_t theta;
  bool keep_matches;

  [[nodiscard]] __host__ __device__ bool operator()(std::uint64_t hash) const noexcept
  {
    if (hash == 0 || hash >= theta) return false;
    std::size_t first = 0;
    std::size_t count = other_size;
    while (count != 0) {
      const std::size_t step = count / 2;
      const std::size_t it   = first + step;
      if (other[it] < hash) {
        first = it + 1;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    const bool found = first != other_size && other[first] == hash;
    return found == keep_matches;
  }
};

}  // namespace datasketches::cuda::detail::theta
