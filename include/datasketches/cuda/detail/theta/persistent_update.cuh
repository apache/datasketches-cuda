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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda/ptx>

#include <cuda_runtime.h>

#include <datasketches/cuda/detail/theta/policy.cuh>

namespace datasketches::cuda::detail::theta {

struct device_state {
  std::uint64_t theta;
  std::uint64_t count;
  std::uint32_t empty;
};

struct persistent_update_config {
  // These are deliberately internal so benchmark experiments can change the
  // kernel geometry without committing to a public policy interface.
  static constexpr int max_block_threads          = 1024;
  static constexpr int target_blocks_per_sm       = 1;
  static constexpr int theta_refresh_warp_rounds  = 128;
  static constexpr std::size_t retained_capacity  = 4096;
  static constexpr std::size_t block_output_slots = retained_capacity + 1;
  static constexpr std::size_t max_load_percent   = 80;
  static constexpr int radix_bits                 = 8;
  static constexpr int radix_buckets              = 1 << radix_bits;
};

static_assert(persistent_update_config::retained_capacity == (std::size_t{1} << default_lg_k));

inline constexpr std::uint64_t empty_key = UINT64_MAX;
static_assert(empty_key > max_theta);

__device__ __forceinline__ std::uint32_t atomic_add_block(std::uint32_t* address,
                                                          std::uint32_t value) noexcept
{
  return atomicAdd_block(address, value);
}

__device__ __forceinline__ std::uint32_t atomic_load_block(std::uint32_t* address) noexcept
{
  return __nv_atomic_load_n(address, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK);
}

__device__ __forceinline__ std::uint32_t atomic_exchange_block(std::uint32_t* address,
                                                               std::uint32_t value) noexcept
{
  return atomicExch_block(address, value);
}

__device__ __forceinline__ std::uint64_t atomic_load_block(std::uint64_t* address) noexcept
{
  return __nv_atomic_load_n(
    reinterpret_cast<unsigned long long*>(address), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK);
}

__device__ __forceinline__ std::uint64_t atomic_load_device(std::uint64_t* address) noexcept
{
  return __nv_atomic_load_n(
    reinterpret_cast<unsigned long long*>(address), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
}

__device__ __forceinline__ void atomic_min_device(std::uint64_t* address,
                                                  std::uint64_t value) noexcept
{
  __nv_atomic_min(reinterpret_cast<unsigned long long*>(address),
                  static_cast<unsigned long long>(value),
                  __NV_ATOMIC_RELAXED,
                  __NV_THREAD_SCOPE_DEVICE);
}

struct identity_hash {
  using argument_type = std::uint64_t;
  using result_type   = std::uint64_t;

  [[nodiscard]] __host__ __device__ constexpr result_type operator()(
    argument_type key) const noexcept
  {
    return key;
  }
};

inline constexpr std::size_t histogram_bytes =
  persistent_update_config::radix_buckets * sizeof(std::uint32_t);

struct persistent_control {
  std::uint32_t count;
  std::uint32_t output_count;
  std::uint32_t has_boundary;
  std::uint32_t stop;
  std::uint32_t finished_warps;
  std::uint32_t selection_rank;
  std::uint64_t theta;
  std::uint64_t selection_prefix;
};

__host__ __device__ inline constexpr std::size_t align_up(std::size_t value,
                                                          std::size_t alignment) noexcept
{
  return (value + alignment - 1) / alignment * alignment;
}

[[nodiscard]] __host__ __device__ inline constexpr std::size_t histogram_offset(
  std::size_t set_slots) noexcept
{
  return align_up(set_slots * sizeof(std::uint64_t), alignof(std::uint32_t));
}

[[nodiscard]] __host__ __device__ inline constexpr std::size_t control_offset(
  std::size_t set_slots) noexcept
{
  return align_up(histogram_offset(set_slots) + histogram_bytes, alignof(persistent_control));
}

[[nodiscard]] inline constexpr std::size_t required_shared_bytes(std::size_t set_slots) noexcept
{
  return control_offset(set_slots) + sizeof(persistent_control);
}

[[nodiscard]] inline constexpr std::size_t max_occupied(std::size_t set_slots) noexcept
{
  return set_slots * persistent_update_config::max_load_percent / 100;
}

__device__ __forceinline__ std::uint32_t initial_slot(std::uint64_t key,
                                                      std::uint32_t slot_count) noexcept
{
  auto bits = static_cast<std::uint32_t>(key) ^ static_cast<std::uint32_t>(key >> 32);
  bits *= 0x9e3779b9U;
  return __umulhi(bits, slot_count);
}

__device__ __forceinline__ bool insert_into_set(std::uint64_t* slots,
                                                std::uint32_t slot_count,
                                                std::uint64_t key) noexcept
{
  auto slot = initial_slot(key, slot_count);
  for (std::uint32_t probe = 0; probe < slot_count; ++probe) {
    auto* slot_address = slots + slot;
    auto observed      = atomic_load_block(slot_address);
    if (observed == key) { return false; }
    if (observed == empty_key) {
      auto expected = empty_key;
      if (__nv_atomic_compare_exchange_n(reinterpret_cast<unsigned long long*>(slot_address),
                                         reinterpret_cast<unsigned long long*>(&expected),
                                         static_cast<unsigned long long>(key),
                                         false,
                                         __NV_ATOMIC_RELAXED,
                                         __NV_ATOMIC_RELAXED,
                                         __NV_THREAD_SCOPE_BLOCK)) {
        return true;
      }
      if (expected == key) { return false; }
    }
    if (++slot == slot_count) { slot = 0; }
  }
  return false;
}

__device__ inline void clear_set(std::uint64_t* slots, std::size_t slot_count)
{
  for (std::size_t slot = threadIdx.x; slot < slot_count; slot += blockDim.x) {
    slots[slot] = empty_key;
  }
}

__device__ inline void compact_set(const std::uint64_t* slots,
                                   std::size_t set_slots,
                                   std::uint64_t* output,
                                   persistent_control& control,
                                   bool apply_theta)
{
  constexpr unsigned int full_mask = 0xffffffffu;
  const unsigned int lane          = threadIdx.x % warpSize;

  if (threadIdx.x == 0) { control.output_count = 0; }
  __syncthreads();

  for (std::size_t base = 0; base < set_slots; base += blockDim.x) {
    const auto slot       = base + threadIdx.x;
    const auto value      = slot < set_slots ? slots[slot] : empty_key;
    const bool emit       = value != empty_key && (!apply_theta || value < control.theta);
    const auto mask       = __ballot_sync(full_mask, emit);
    const auto warp_count = static_cast<std::uint32_t>(__popc(mask));

    std::uint32_t warp_base = 0;
    if (lane == 0 && warp_count != 0) {
      warp_base = atomic_add_block(&control.output_count, warp_count);
    }
    warp_base = __shfl_sync(full_mask, warp_base, 0);

    const auto lower_lanes = lane == 0 ? 0U : ((1U << lane) - 1U);
    if (emit) { output[warp_base + __popc(mask & lower_lanes)] = value; }
  }
  __syncthreads();
}

// Finds the exact rank-k key without permuting the open-addressing table.
// Each pass narrows the active prefix by one byte.
__device__ inline void select_local_boundary(const std::uint64_t* slots,
                                             std::size_t set_slots,
                                             std::uint32_t* histogram,
                                             persistent_control& control)
{
  if (threadIdx.x == 0) {
    control.selection_prefix = 0;
    control.selection_rank   = persistent_update_config::retained_capacity;
  }
  __syncthreads();

  for (int shift = 64 - persistent_update_config::radix_bits; shift >= 0;
       shift -= persistent_update_config::radix_bits) {
    for (int bucket = threadIdx.x; bucket < persistent_update_config::radix_buckets;
         bucket += blockDim.x) {
      histogram[bucket] = 0;
    }
    __syncthreads();

    const auto prefix_mask = shift == 64 - persistent_update_config::radix_bits
                               ? std::uint64_t{0}
                               : (UINT64_MAX << (shift + persistent_update_config::radix_bits));
    for (std::size_t slot = threadIdx.x; slot < set_slots; slot += blockDim.x) {
      const auto value = slots[slot];
      if (value != empty_key && (value & prefix_mask) == control.selection_prefix) {
        const auto digit = static_cast<std::uint32_t>(
          (value >> shift) & (persistent_update_config::radix_buckets - 1));
        atomicAdd(histogram + digit, 1U);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      std::uint32_t preceding = 0;
      for (std::uint32_t digit = 0; digit < persistent_update_config::radix_buckets; ++digit) {
        const auto bucket_count = histogram[digit];
        if (control.selection_rank < preceding + bucket_count) {
          control.selection_rank -= preceding;
          control.selection_prefix |= static_cast<std::uint64_t>(digit) << shift;
          break;
        }
        preceding += bucket_count;
      }
    }
    __syncthreads();
  }
}

// Spill the selected bottom-k into the block's global output segment before
// collectively clearing and rebuilding the shared set.
__device__ inline void reduce_local_set(std::uint64_t* slots,
                                        std::size_t set_slots,
                                        std::uint32_t* histogram,
                                        persistent_control& control,
                                        device_state& state,
                                        std::uint64_t* workspace)
{
  select_local_boundary(slots, set_slots, histogram, control);

  if (threadIdx.x == 0) {
    control.theta        = control.selection_prefix;
    control.has_boundary = 1;
    atomic_min_device(&state.theta, control.theta);
  }
  __syncthreads();

  compact_set(slots, set_slots, workspace, control, true);
  clear_set(slots, set_slots);
  __syncthreads();
  for (std::size_t i = threadIdx.x; i < persistent_update_config::retained_capacity;
       i += blockDim.x) {
    (void)insert_into_set(slots, static_cast<std::uint32_t>(set_slots), workspace[i]);
  }
  __syncthreads();
  if (threadIdx.x == 0) { control.count = persistent_update_config::retained_capacity; }
  __syncthreads();
}

template <class KeyIt, class Hasher>
__global__ __launch_bounds__(
  persistent_update_config::max_block_threads,
  persistent_update_config::
    target_blocks_per_sm) void persistent_update_kernel(KeyIt keys,
                                                        std::size_t num_keys,
                                                        Hasher hasher,
                                                        device_state* state,
                                                        std::uint64_t* block_output,
                                                        std::size_t set_slot_count,
                                                        std::size_t load_limit)
{
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto* slots = reinterpret_cast<std::uint64_t*>(dynamic_shared);
  auto* histogram =
    reinterpret_cast<std::uint32_t*>(dynamic_shared + histogram_offset(set_slot_count));
  auto* control =
    reinterpret_cast<persistent_control*>(dynamic_shared + control_offset(set_slot_count));

  auto* workspace = block_output + static_cast<std::size_t>(blockIdx.x) *
                                     persistent_update_config::block_output_slots;

  clear_set(slots, set_slot_count);
  if (threadIdx.x == 0) {
    control->count            = 0;
    control->output_count     = 0;
    control->has_boundary     = 0;
    control->stop             = 0;
    control->finished_warps   = 0;
    control->selection_rank   = 0;
    control->theta            = atomic_load_device(&state->theta);
    control->selection_prefix = 0;
  }
  __syncthreads();

  const std::size_t base_count = num_keys / gridDim.x;
  const std::size_t remainder  = num_keys % gridDim.x;
  const auto block_index       = static_cast<std::size_t>(blockIdx.x);
  const std::size_t begin =
    block_index * base_count + (block_index < remainder ? block_index : remainder);
  const std::size_t local_count =
    base_count + (static_cast<std::size_t>(blockIdx.x) < remainder ? 1 : 0);
  constexpr unsigned int full_mask = 0xffffffffu;
  const bool warp_leader           = ::cuda::ptx::elect_sync(full_mask);
  const unsigned int leader_lane =
    static_cast<unsigned int>(__ffs(static_cast<int>(__ballot_sync(full_mask, warp_leader))) - 1);
  const unsigned int lane          = threadIdx.x % warpSize;
  const unsigned int warp          = threadIdx.x / warpSize;
  const unsigned int num_warps     = blockDim.x / warpSize;
  const std::size_t warp_base      = local_count / num_warps;
  const std::size_t warp_remainder = local_count % num_warps;
  std::size_t round = begin + warp * warp_base + (warp < warp_remainder ? warp : warp_remainder);
  const std::size_t warp_count =
    warp_base + (static_cast<std::size_t>(warp) < warp_remainder ? 1 : 0);
  const std::size_t warp_end = round + warp_count;

  while (true) {
    if (threadIdx.x == 0) {
      control->stop           = 0;
      control->finished_warps = 0;
    }
    __syncthreads();

    auto warp_theta          = control->theta;
    std::uint32_t warp_round = 0;
    while (round < warp_end) {
      std::uint32_t stop = 0;
      if (warp_leader) {
        stop = atomic_load_block(&control->stop);
        if (stop == 0 && warp_round % persistent_update_config::theta_refresh_warp_rounds == 0) {
          const auto global_theta = atomic_load_device(&state->theta);
          warp_theta              = global_theta < warp_theta ? global_theta : warp_theta;
        }
      }
      stop       = __shfl_sync(full_mask, stop, leader_lane);
      warp_theta = __shfl_sync(full_mask, warp_theta, leader_lane);
      if (stop != 0) { break; }

      const auto index  = round + lane;
      const bool active = index < warp_end;
      const auto hash   = active ? static_cast<std::uint64_t>(hasher(keys[index])) : empty_key;
      const bool inserted =
        active && hash != 0 && hash < warp_theta && hash != empty_key &&
        insert_into_set(slots, static_cast<std::uint32_t>(set_slot_count), hash);
      const auto inserted_mask = __ballot_sync(full_mask, inserted);
      if (warp_leader && inserted_mask != 0) {
        const auto inserted_count = static_cast<std::uint32_t>(__popc(inserted_mask));
        const auto old_count      = atomic_add_block(&control->count, inserted_count);
        if (old_count + inserted_count >= load_limit) { atomic_exchange_block(&control->stop, 1U); }
      }

      const auto remaining = warp_end - round;
      round += remaining < static_cast<std::size_t>(warpSize) ? remaining
                                                              : static_cast<std::size_t>(warpSize);
      ++warp_round;
    }

    if (warp_leader && round == warp_end) { atomic_add_block(&control->finished_warps, 1U); }
    __syncthreads();

    const auto reduce = __syncthreads_or(threadIdx.x == 0 && control->finished_warps != num_warps);
    if (reduce == 0) { break; }

    reduce_local_set(slots, set_slot_count, histogram, *control, *state, workspace);
  }

  const auto reduce = __syncthreads_or(
    threadIdx.x == 0 && control->count > persistent_update_config::retained_capacity);
  if (reduce != 0) {
    reduce_local_set(slots, set_slot_count, histogram, *control, *state, workspace);
  }

  for (std::size_t i = threadIdx.x; i < persistent_update_config::block_output_slots;
       i += blockDim.x) {
    workspace[i] = empty_key;
  }
  __syncthreads();
  compact_set(slots, set_slot_count, workspace, *control, false);

  if (threadIdx.x == 0 && control->has_boundary != 0) {
    workspace[control->output_count] = control->theta;
  }
}

__global__ inline void finalize_update_kernel(const std::uint64_t* unique,
                                              const std::uint64_t* unique_count,
                                              std::uint64_t* retained,
                                              device_state* state,
                                              bool saw_input)
{
  __shared__ std::uint64_t valid_count;
  __shared__ std::uint64_t next_theta;

  if (threadIdx.x == 0) {
    valid_count = *unique_count;
    if (valid_count != 0 && unique[valid_count - 1] == empty_key) { --valid_count; }
    next_theta = state->theta;
    if (valid_count > persistent_update_config::retained_capacity) {
      const auto candidate_theta = unique[persistent_update_config::retained_capacity];
      next_theta                 = candidate_theta < next_theta ? candidate_theta : next_theta;
    }
    state->count = valid_count < persistent_update_config::retained_capacity
                     ? valid_count
                     : persistent_update_config::retained_capacity;
    state->theta = next_theta;
    if (saw_input) { state->empty = 0; }
  }
  __syncthreads();

  for (std::size_t i = threadIdx.x; i < persistent_update_config::retained_capacity;
       i += blockDim.x) {
    retained[i] = i < state->count ? unique[i] : empty_key;
  }
}

}  // namespace datasketches::cuda::detail::theta
