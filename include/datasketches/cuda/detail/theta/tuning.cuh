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

namespace datasketches::cuda::detail::theta {

struct tuning {
  static constexpr ::cuda::std::int32_t block_threads    = 256;
  static constexpr ::cuda::std::int32_t items_per_thread = 8;
  static constexpr ::cuda::std::size_t filter_slots      = 1024;
  static constexpr ::cuda::std::size_t tile_keys         = block_threads * items_per_thread;

  // Amortize launches while bounding sort and screen scratch independently of input size.
  static constexpr ::cuda::std::size_t chunk_target_multiple = 16;
  static constexpr ::cuda::std::size_t min_chunk_keys        = ::cuda::std::size_t{1} << 20;
  static constexpr ::cuda::std::size_t min_survivor_budget   = ::cuda::std::size_t{1} << 22;
  static constexpr ::cuda::std::size_t max_chunk_growth      = 64;
  static constexpr double budget_margin                      = 1.1;
  static constexpr double duplicate_hedge                    = 2.0;

  static ::cuda::std::uint32_t grid_size(::cuda::std::size_t count) noexcept
  {
    constexpr auto max_blocks = ::cuda::std::size_t{0x7fffffff};
    const auto blocks         = (count + tile_keys - 1) / tile_keys;
    return static_cast<::cuda::std::uint32_t>(blocks < max_blocks ? blocks : max_blocks);
  }
};

}  // namespace datasketches::cuda::detail::theta
