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

#include <cuda/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/utility>

#include <cuda_runtime.h>

#include <cub/warp/warp_scan.cuh>

#include <datasketches/cuda/detail/theta/policy.cuh>
#include <datasketches/cuda/detail/theta/tuning.cuh>

namespace datasketches::cuda::detail::theta {

// Hash, screen, and compact without materializing one hash per input key.
// Warp and block duplicate filters are best-effort: only exact matches are
// dropped; missed duplicates are removed by the subsequent sort/unique.
// Count all survivors but guard writes by capacity so the caller can retry.
// Each warp reserves one output run for all its items, amortizing atomics
// without a block-wide synchronization at each item.
template <class KeyIt, class Hasher>
__global__ void screen_kernel(KeyIt keys,
                              ::cuda::std::size_t num_keys,
                              Hasher hasher,
                              ::cuda::std::uint64_t theta,
                              ::cuda::std::uint64_t* __restrict__ out,
                              ::cuda::std::size_t capacity,
                              ::cuda::std::uint64_t* __restrict__ out_count)
{
  constexpr ::cuda::std::uint32_t warp_width = 32;
  constexpr ::cuda::std::uint32_t full_mask  = 0xffffffffu;
  const ::cuda::std::uint32_t lane           = threadIdx.x % warp_width;

  using warp_scan = cub::WarpScan<::cuda::std::int32_t>;
  __shared__ typename warp_scan::TempStorage scan_storage[tuning::block_threads / warp_width];

  // A slot contains the entire hash; it publishes no other data. Relaxed
  // atomics suffice: stale observations can only emit another duplicate.
  using filter_cell = ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_block>;
  __shared__ ::cuda::std::uint64_t filter[tuning::filter_slots];
  for (auto i = static_cast<::cuda::std::size_t>(threadIdx.x); i < tuning::filter_slots;
       i += tuning::block_threads) {
    filter[i] = 0;
  }
  __syncthreads();

  // A grid from tuning::grid_size covers the input in one step; the stride loop
  // keeps the kernel correct for any smaller grid a caller might pass.
  const auto stride = static_cast<::cuda::std::size_t>(gridDim.x) * tuning::tile_keys;

  for (auto base = static_cast<::cuda::std::size_t>(blockIdx.x) * tuning::tile_keys;
       base < num_keys;
       base += stride) {
    ::cuda::std::uint64_t hashes[tuning::items_per_thread];
    ::cuda::std::int32_t mine = 0;

    ::cuda::static_for<tuning::items_per_thread>([&](auto i) {
      // strided within the tile so every load step stays fully coalesced
      const auto idx = base + i * tuning::block_threads + threadIdx.x;
      hashes[i]      = 0;
      const ::cuda::std::uint64_t hash =
        idx < num_keys ? hasher(keys[idx]) : ::cuda::std::uint64_t{0};
      const bool survives = screen_hash{theta}(hash);

      // Stage one: one lane per distinct hash in this warp continues.
      const auto peers = __match_any_sync(full_mask, survives ? hash : ::cuda::std::uint64_t{0});
      const bool leader =
        survives && static_cast<::cuda::std::uint32_t>(__ffs(static_cast<int>(peers)) - 1) == lane;

      // Stage two: drop it if this block emitted the same hash recently.
      if (leader) {
        filter_cell cell{filter[hash % tuning::filter_slots]};
        if (cell.load(::cuda::memory_order_relaxed) != hash) {
          cell.store(hash, ::cuda::memory_order_relaxed);
          hashes[i] = hash;
          ++mine;
        }
      }
    });

    // One atomic per warp: the warp scan yields each lane's offset and the total.
    ::cuda::std::int32_t scan       = 0;
    ::cuda::std::int32_t warp_total = 0;
    warp_scan(scan_storage[threadIdx.x / warp_width]).ExclusiveSum(mine, scan, warp_total);
    if (warp_total == 0) { continue; }

    ::cuda::std::uint64_t warp_base = 0;
    if (lane == 0) {
      warp_base =
        ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device>{*out_count}
          .fetch_add(static_cast<::cuda::std::uint64_t>(warp_total), ::cuda::memory_order_relaxed);
    }
    auto cursor = __shfl_sync(full_mask, warp_base, 0) + static_cast<::cuda::std::uint64_t>(scan);
    ::cuda::static_for<tuning::items_per_thread>([&](auto i) {
      if (hashes[i] != 0) {
        if (cursor < capacity) { out[cursor] = hashes[i]; }
        ++cursor;
      }
    });
  }
}

}  // namespace datasketches::cuda::detail::theta
