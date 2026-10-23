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

#include <algorithm>
#include <cstdint>
#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <datasketches/cuda/theta.hpp>

#include <nvbench/nvbench.cuh>

namespace {

using key_type = ::cuda::std::uint64_t;

//! @brief Generates keys with a controlled distinct count and arrival locality.
//!
//! Every key appears `count / distinct` times, with one extra occurrence for
//! the first `count % distinct` keys. Scattered and grouped layouts contain
//! exactly the same multiset: scattered repeats are `distinct` positions apart,
//! while grouped repeats are adjacent.
//!
//! Both axes matter and they are not interchangeable. Duplicate-heavy input
//! keeps theta high for longer regardless of ordering, which exercises the
//! chunking path. Locality separately decides whether repeats are visible to any
//! kernel that only sees a bounded window of consecutive keys, so a benchmark
//! that only ever generates scattered repeats cannot distinguish an optimization that
//! exploits locality from one that does nothing.
struct generate_key {
  ::cuda::std::uint64_t distinct;
  ::cuda::std::uint64_t repeats;
  ::cuda::std::uint64_t extra;
  ::cuda::std::uint64_t grouped_prefix;
  bool grouped;

  __host__ __device__ key_type operator()(::cuda::std::uint64_t index) const noexcept
  {
    const auto key              = !grouped
                                    ? index % distinct
                                    : (index < grouped_prefix ? index / (repeats + 1)
                                                              : extra + (index - grouped_prefix) / repeats);
    ::cuda::std::uint64_t value = key + 0x9e3779b97f4a7c15ULL;
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
  }
};

thrust::device_vector<key_type> make_keys(std::size_t count,
                                          ::cuda::std::uint64_t distinct,
                                          bool grouped = false)
{
  const auto repeats = count / distinct;
  const auto extra   = count % distinct;
  thrust::device_vector<key_type> keys(count);
  thrust::transform(thrust::device,
                    ::cuda::counting_iterator<::cuda::std::uint64_t>{0},
                    ::cuda::counting_iterator<::cuda::std::uint64_t>{count},
                    keys.begin(),
                    generate_key{distinct, repeats, extra, extra * (repeats + 1), grouped});
  return keys;
}

//! @brief Cost of filling an empty sketch.
//!
//! The sketch starts at maximum theta and enters estimation mode only if the
//! batch contains more than k distinct keys.
void theta_update_cold(nvbench::state& state)
{
  const auto num_keys     = static_cast<std::size_t>(state.get_int64("Keys"));
  const auto distinct_pct = state.get_int64("DistinctPct");
  const auto lg_k         = static_cast<std::uint8_t>(state.get_int64("LgK"));
  const auto grouped      = state.get_string("Layout") == "grouped";
  const auto distinct     = std::max<std::uint64_t>(1, num_keys * distinct_pct / 100);

  if (grouped && distinct == num_keys) {
    state.skip("Distinct keys have the same scattered and grouped layout");
    return;
  }
  const auto keys = make_keys(num_keys, distinct, grouped);
  auto mr         = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  state.add_element_count(num_keys, "Keys");
  state.add_global_memory_reads<key_type>(num_keys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    ::cuda::stream_ref stream{launch.get_stream()};
    datasketches::cuda::theta_sketch<key_type> sketch(stream, mr, lg_k);
    sketch.update(stream, keys.begin(), keys.end());
  });
}

//! @brief Cost of feeding a fixed key count through many small update() calls.
//!
//! Each call carries a fixed cost in launches and synchronizations that does not
//! shrink with the batch, so this separates per-call overhead from streaming
//! throughput. Sweeping Batch at a fixed Keys shows where the two cross over.
void theta_update_batched(nvbench::state& state)
{
  const auto num_keys = static_cast<std::size_t>(state.get_int64("Keys"));
  const auto batch    = static_cast<std::size_t>(state.get_int64("Batch"));
  const auto lg_k     = static_cast<std::uint8_t>(state.get_int64("LgK"));

  const auto keys = make_keys(num_keys, num_keys);
  auto mr         = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  state.add_element_count(num_keys, "Keys");
  state.add_global_memory_reads<key_type>(num_keys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    ::cuda::stream_ref stream{launch.get_stream()};
    datasketches::cuda::theta_sketch<key_type> sketch(stream, mr, lg_k);
    for (std::size_t offset = 0; offset < num_keys; offset += batch) {
      const auto end = std::min(offset + batch, num_keys);
      sketch.update(stream, keys.begin() + offset, keys.begin() + end);
    }
  });
}

//! @brief Cost of repeatedly merging two identical saturated sketches.
void theta_merge(nvbench::state& state)
{
  const auto num_keys = static_cast<std::size_t>(state.get_int64("Keys"));
  const auto lg_k     = static_cast<std::uint8_t>(state.get_int64("LgK"));

  const auto keys = make_keys(num_keys, num_keys);
  auto mr         = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  // Both sketches contain the same hashes, so this measures full-overlap union.
  // Repeated merges preserve their state and perform the same work each time.
  ::cuda::stream setup_stream{::cuda::devices[0]};
  datasketches::cuda::theta_sketch<key_type> other(setup_stream, mr, lg_k);
  other.update(setup_stream, keys.begin(), keys.end());
  datasketches::cuda::theta_sketch<key_type> sketch(setup_stream, mr, lg_k);
  sketch.update(setup_stream, keys.begin(), keys.end());

  state.add_element_count(std::size_t{1} << lg_k, "Retained");

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    ::cuda::stream_ref stream{launch.get_stream()};
    sketch.merge(stream, other);
  });
}

}  // namespace

NVBENCH_BENCH(theta_update_cold)
  .set_name("theta_update_cold")
  .add_int64_axis("Keys", {1 << 20, 1 << 24, 100'000'000})
  .add_int64_axis("DistinctPct", {100, 1})
  .add_string_axis("Layout", {"scattered", "grouped"})
  .add_int64_axis("LgK", {12, 20});

NVBENCH_BENCH(theta_update_batched)
  .set_name("theta_update_batched")
  .add_int64_axis("Keys", {1 << 24})
  .add_int64_axis("Batch", {1 << 16, 1 << 20, 1 << 24})
  .add_int64_axis("LgK", {12});

NVBENCH_BENCH(theta_merge)
  .set_name("theta_merge_full_overlap")
  .add_int64_axis("Keys", {1 << 22})
  .add_int64_axis("LgK", {12, 16});
