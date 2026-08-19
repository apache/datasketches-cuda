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
#include <cuda/memory_pool>
#include <cuda/stream>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <datasketches/cuda/theta.hpp>

#include <nvbench/nvbench.cuh>

namespace {

using key_type = std::uint64_t;

//! @brief Generates keys with a controlled distinct count and arrival locality.
//!
//! `distinct` sets how many distinct values appear. `run` sets how many
//! consecutive positions each one occupies before the next: `run == 1` scatters
//! repeats `distinct` positions apart, while a large `run` makes them arrive in
//! blocks the way sorted or grouped input does.
//!
//! Both axes matter and they are not interchangeable. Duplicate-heavy input
//! keeps theta high for longer regardless of ordering, which exercises the
//! chunking path. Locality separately decides whether repeats are visible to any
//! kernel that only sees a bounded window of consecutive keys, so a benchmark
//! that only ever generates `run == 1` cannot distinguish an optimization that
//! exploits locality from one that does nothing.
struct generate_key {
  std::uint64_t distinct;
  std::uint64_t run;

  __host__ __device__ key_type operator()(std::uint64_t index) const noexcept
  {
    std::uint64_t value = ((index / run) % distinct) + 0x9e3779b97f4a7c15ULL;
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
  }
};

thrust::device_vector<key_type> make_keys(std::size_t count,
                                          std::uint64_t distinct,
                                          std::uint64_t run = 1)
{
  thrust::device_vector<key_type> keys(count);
  thrust::transform(thrust::counting_iterator<std::uint64_t>(0),
                    thrust::counting_iterator<std::uint64_t>(count),
                    keys.begin(),
                    generate_key{distinct, run});
  return keys;
}

//! @brief Cost of filling an empty sketch.
//!
//! A sketch entering update() with theta at its maximum rejects nothing, so this
//! measures the path where update() has to tighten theta partway through the
//! batch rather than screening against an already-small theta.
void theta_update_cold(nvbench::state& state)
{
  const auto num_keys     = static_cast<std::size_t>(state.get_int64("Keys"));
  const auto distinct_pct = state.get_int64("DistinctPct");
  const auto lg_k         = static_cast<std::uint8_t>(state.get_int64("LgK"));
  const auto run          = static_cast<std::uint64_t>(state.get_int64("Run"));
  const auto distinct     = std::max<std::uint64_t>(1, num_keys * distinct_pct / 100);

  const auto keys = make_keys(num_keys, distinct, run);
  auto mr         = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  state.add_element_count(num_keys, "Keys");
  state.add_global_memory_reads<key_type>(num_keys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    ::cuda::stream_ref stream{launch.get_stream()};
    datasketches::cuda::theta_sketch<key_type> sketch(stream, mr, lg_k);
    sketch.update(stream, keys.begin(), keys.end());
  });
}

//! @brief Cost of updating a sketch that has already reached its nominal k.
//!
//! This is the steady state for a streaming workload: theta is small, nearly
//! every key is rejected by the screen, and throughput is set by how fast the
//! sketch can hash and reject.
void theta_update_warm(nvbench::state& state)
{
  const auto num_keys     = static_cast<std::size_t>(state.get_int64("Keys"));
  const auto distinct_pct = state.get_int64("DistinctPct");
  const auto lg_k         = static_cast<std::uint8_t>(state.get_int64("LgK"));
  const auto run          = static_cast<std::uint64_t>(state.get_int64("Run"));
  const auto distinct     = std::max<std::uint64_t>(1, num_keys * distinct_pct / 100);

  const auto prime_keys = std::max<std::size_t>(1, num_keys / 10);
  const auto keys       = make_keys(num_keys, distinct, run);
  const auto prime      = make_keys(prime_keys, std::max<std::uint64_t>(1, distinct / 10), run);
  auto mr               = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  // Saturate the sketch outside the timed region. Re-running the measured update
  // against the same sketch is the steady state being measured: theta is already
  // small, so repeated updates neither grow the retained set nor move theta.
  ::cuda::stream setup_stream{::cuda::devices[0]};
  datasketches::cuda::theta_sketch<key_type> sketch(setup_stream, mr, lg_k);
  sketch.update(setup_stream, prime.begin(), prime.end());
  setup_stream.sync();

  state.add_element_count(num_keys, "Keys");
  state.add_global_memory_reads<key_type>(num_keys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    ::cuda::stream_ref stream{launch.get_stream()};
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

//! @brief Cost of merging two saturated sketches.
void theta_merge(nvbench::state& state)
{
  const auto num_keys = static_cast<std::size_t>(state.get_int64("Keys"));
  const auto lg_k     = static_cast<std::uint8_t>(state.get_int64("LgK"));

  const auto left  = make_keys(num_keys, num_keys);
  const auto right = make_keys(num_keys, num_keys);
  auto mr          = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  // Both sides are built outside the timed region. Merging the same other sketch
  // repeatedly is stable once the first merge has run, so the measured call does
  // the same work on every iteration.
  ::cuda::stream setup_stream{::cuda::devices[0]};
  datasketches::cuda::theta_sketch<key_type> other(setup_stream, mr, lg_k);
  other.update(setup_stream, right.begin(), right.end());
  datasketches::cuda::theta_sketch<key_type> sketch(setup_stream, mr, lg_k);
  sketch.update(setup_stream, left.begin(), left.end());
  setup_stream.sync();

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
  .add_int64_axis("Run", {1, 2048})
  .add_int64_axis("LgK", {12, 20});

NVBENCH_BENCH(theta_update_warm)
  .set_name("theta_update_warm")
  .add_int64_axis("Keys", {1 << 20, 1 << 24, 100'000'000})
  .add_int64_axis("DistinctPct", {100, 1})
  .add_int64_axis("Run", {1, 2048})
  .add_int64_axis("LgK", {12, 20});

NVBENCH_BENCH(theta_update_batched)
  .set_name("theta_update_batched")
  .add_int64_axis("Keys", {1 << 24})
  .add_int64_axis("Batch", {1 << 16, 1 << 20, 1 << 24})
  .add_int64_axis("LgK", {12});

NVBENCH_BENCH(theta_merge)
  .set_name("theta_merge")
  .add_int64_axis("Keys", {1 << 22})
  .add_int64_axis("LgK", {12, 16});
