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

#include <cuda_runtime.h>

namespace datasketches::cuda::detail::theta {

//! @brief Threads per block used by @ref screen_kernel.
inline constexpr int screen_block_threads = 256;

//! @brief Keys hashed per thread per grid-stride step by @ref screen_kernel.
inline constexpr int screen_items_per_thread = 8;

//! @brief Slots in the block-local duplicate filter of @ref screen_kernel.
//!
//! 1024 slots of 8 bytes is 8 KiB per block. Thread-count limits bind before
//! shared memory does at that size on every architecture in the CUDA
//! per-compute-capability table, so the filter costs no occupancy. Larger
//! filters measured no better; the working set that fits is not the limiting
//! factor, warp-level collapse is.
inline constexpr ::cuda::std::size_t screen_filter_slots = 1024;

//! @brief Keys handled by one block per grid-stride step of @ref screen_kernel.
inline constexpr ::cuda::std::size_t screen_tile_keys =
  static_cast<::cuda::std::size_t>(screen_block_threads) * screen_items_per_thread;

//! @brief Returns the grid size to launch @ref screen_kernel with.
//!
//! Sized to cover the input in a single grid-stride step. Deriving the grid from
//! the input rather than from the device keeps the launch free of any hardware
//! query or occupancy constant, and measures no slower than an SM-count-derived
//! grid tuned for a particular part.
//!
//! @param num_keys Number of keys the launch will process
//! @return Number of blocks to launch, at least one
[[nodiscard]] inline unsigned int screen_grid_size(::cuda::std::size_t num_keys) noexcept
{
  constexpr auto max_blocks = ::cuda::std::size_t{0x7fffffff};
  const auto blocks         = (num_keys + screen_tile_keys - 1) / screen_tile_keys;
  return static_cast<unsigned int>(blocks == 0 ? 1 : (blocks < max_blocks ? blocks : max_blocks));
}

//! @brief Hashes a key range, screens it against theta, and compacts the survivors.
//!
//! Fuses what would otherwise be a `cub::DeviceTransform` pass writing one hash
//! per input key followed by a `cub::DeviceSelect::If` pass reading it back. The
//! intermediate array is never materialized, which halves the memory traffic of
//! the screen.
//!
//! Survivors then pass a two-stage duplicate filter before being emitted. Both
//! stages are best-effort: a missed duplicate only means one extra entry reaches
//! the sort, which removes it anyway, so correctness never depends on either.
//!
//! The first stage collapses duplicates that are live in the same warp at the
//! same instant. `__match_any_sync` groups lanes holding equal hashes and keeps
//! one per group. This is where nearly all of the benefit comes from, because
//! input whose duplicates are adjacent (sorted or grouped data) puts them in the
//! same warp.
//!
//! The second stage is a direct-mapped table indexed by `hash % slots`, which
//! catches duplicates separated in time within a block. A slot holds a full
//! 64-bit hash and a key is dropped only on an exact match, so two hashes
//! colliding on a slot merely evict each other and both are emitted; a collision
//! can never drop a distinct value. Being direct-mapped rather than
//! open-addressed is what keeps it safe: cost is one load and one store no
//! matter how full it is, with no probe chain to grow and no rehashing, so an
//! oversubscribed filter simply stops hitting instead of falling off a cliff.
//!
//! Compaction is warp-local: each thread counts its own survivors, a `__shfl_up`
//! scan turns those counts into per-thread offsets, and one lane claims the
//! warp's whole run with a single atomic. A block-wide scan would instead
//! synchronize every thread once per @ref screen_block_threads keys, which
//! dominates the kernel whenever theta is small enough to reject most keys, and
//! that is the steady state for a Theta sketch that has reached its nominal k.
//!
//! Voting per item rather than per thread would be just as bad in the other
//! direction: at a pass rate of a few percent most items have at least one
//! surviving lane in the warp, so each item costs an atomic. Aggregating all
//! @ref screen_items_per_thread items into one atomic avoids both failure modes.
//!
//! @note The output order is unspecified. Callers sort the survivors anyway.
//!
//! @tparam KeyIt Random-access iterator over the input keys
//! @tparam Hasher Callable mapping a key to its 64-bit Theta hash
//!
//! @param[in] keys Device range of input keys
//! @param[in] num_keys Number of keys in @p keys
//! @param[in] hasher Theta hash functor
//! @param[in] theta Current sketch theta; hashes must be non-zero and below it
//! @param[out] out Receives the surviving hashes, unordered
//! @param[in,out] out_count Running count of survivors written to @p out
template <class KeyIt, class Hasher>
__global__ void screen_kernel(KeyIt keys,
                              ::cuda::std::size_t num_keys,
                              Hasher hasher,
                              ::cuda::std::uint64_t theta,
                              ::cuda::std::uint64_t* __restrict__ out,
                              unsigned long long* __restrict__ out_count)
{
  constexpr unsigned int warp_width = 32;
  constexpr unsigned int full_mask  = 0xffffffffu;
  const unsigned int lane           = threadIdx.x % warp_width;

  // Concurrent threads may race on a slot. Losing a remembered hash or emitting
  // a duplicate are both harmless, so the race is by design, but it is still a
  // race and the accesses are relaxed atomics rather than plain loads and stores.
  using filter_cell = ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_block>;
  __shared__ ::cuda::std::uint64_t filter[screen_filter_slots];
  for (auto i = static_cast<::cuda::std::size_t>(threadIdx.x); i < screen_filter_slots;
       i += screen_block_threads) {
    filter[i] = 0;
  }
  __syncthreads();

  // A grid from screen_grid_size covers the input in one step; the stride loop
  // keeps the kernel correct for any smaller grid a caller might pass.
  const auto stride = static_cast<::cuda::std::size_t>(gridDim.x) * screen_tile_keys;

  for (auto base = static_cast<::cuda::std::size_t>(blockIdx.x) * screen_tile_keys; base < num_keys;
       base += stride) {
    ::cuda::std::uint64_t hashes[screen_items_per_thread];
    int mine = 0;

#pragma unroll
    for (int i = 0; i < screen_items_per_thread; ++i) {
      // strided within the tile so every load step stays fully coalesced
      const auto idx = base + i * screen_block_threads + threadIdx.x;
      hashes[i]      = 0;
      const ::cuda::std::uint64_t hash =
        idx < num_keys ? hasher(keys[idx]) : ::cuda::std::uint64_t{0};
      const bool survives = (hash != 0 && hash < theta);

      // Stage one: one lane per distinct hash in this warp continues.
      const auto peers = __match_any_sync(full_mask, survives ? hash : ::cuda::std::uint64_t{0});
      const bool leader =
        survives && (__ffs(static_cast<int>(peers)) - 1) == static_cast<int>(lane);

      // Stage two: drop it if this block emitted the same hash recently.
      if (leader) {
        filter_cell cell{filter[hash % screen_filter_slots]};
        if (cell.load(::cuda::memory_order_relaxed) != hash) {
          cell.store(hash, ::cuda::memory_order_relaxed);
          hashes[i] = hash;
          ++mine;
        }
      }
    }

    int scan = mine;
#pragma unroll
    for (unsigned int offset = 1; offset < warp_width; offset <<= 1) {
      const int prev = __shfl_up_sync(full_mask, scan, offset);
      if (lane >= offset) { scan += prev; }
    }
    const int warp_total = __shfl_sync(full_mask, scan, warp_width - 1);
    if (warp_total == 0) { continue; }

    unsigned long long warp_base = 0;
    if (lane == warp_width - 1) {
      warp_base = atomicAdd(out_count, static_cast<unsigned long long>(warp_total));
    }
    warp_base = __shfl_sync(full_mask, warp_base, warp_width - 1);

    auto cursor = warp_base + static_cast<unsigned long long>(scan - mine);
#pragma unroll
    for (int i = 0; i < screen_items_per_thread; ++i) {
      if (hashes[i] != 0) { out[cursor++] = hashes[i]; }
    }
  }
}

}  // namespace datasketches::cuda::detail::theta
