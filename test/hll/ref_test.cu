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
#include <cstddef>
#include <cstdint>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <hll.hpp>

#include <datasketches/cuda/hll.hpp>
#include <datasketches/cuda/hll_ref.cuh>

#include <cooperative_groups.h>

namespace {

constexpr std::size_t register_offset = 40;
constexpr std::size_t flags_offset    = 5;
constexpr std::uint8_t ooo_flag_mask  = 0x10;

struct ref_query_result {
  std::size_t estimate;
  double lower_bound;
  double upper_bound;
  bool empty;
  std::uint8_t lg_k;
  int target_type;
  std::size_t num_registers;
};

template <class Ref>
__global__ void clear_ref_kernel(Ref ref)
{
  ref.clear(cooperative_groups::this_thread_block());
}

template <class Ref, class Key>
__global__ void update_ref_kernel(Ref ref, const Key* keys, std::size_t size)
{
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    ref.update(keys[i]);
  }
}

template <class Ref>
__global__ void estimate_ref_kernel(Ref ref, std::size_t* result)
{
  const auto block    = cooperative_groups::this_thread_block();
  const auto estimate = ref.get_estimate(block);
  if (block.thread_rank() == 0) { *result = estimate; }
}

template <class Ref>
__global__ void query_ref_kernel(Ref ref, std::uint8_t num_std_dev, ref_query_result* result)
{
  const auto block = cooperative_groups::this_thread_block();
  ref_query_result query{};
  query.estimate      = ref.get_estimate(block);
  query.lower_bound   = ref.get_lower_bound(block, num_std_dev);
  query.upper_bound   = ref.get_upper_bound(block, num_std_dev);
  query.empty         = ref.is_empty(block);
  query.lg_k          = ref.get_lg_config_k();
  query.target_type   = static_cast<int>(ref.get_target_type());
  query.num_registers = ref.num_registers();
  if (block.thread_rank() == 0) { *result = query; }
}

template <class Ref, class OtherRef>
__global__ void merge_ref_kernel(Ref destination, OtherRef source)
{
  destination.merge(cooperative_groups::this_thread_block(), source);
}

template <class Key>
__global__ void block_ref_kernel(const Key* keys,
                                 std::size_t size,
                                 std::size_t sketch_bytes,
                                 std::size_t* result)
{
  extern __shared__ ::cuda::std::byte storage[];
  using ref_type = datasketches::cuda::hll_sketch_ref<Key, ::cuda::thread_scope_block>;

  const auto block = cooperative_groups::this_thread_block();
  ref_type ref{::cuda::std::span<::cuda::std::byte>{storage, sketch_bytes}};
  ref.clear(block);
  block.sync();

  for (std::size_t i = block.thread_rank(); i < size; i += block.size()) {
    ref.update(keys[i]);
  }
  block.sync();

  const auto estimate = ref.get_estimate(block);
  if (block.thread_rank() == 0) { *result = estimate; }
}

template <class Key>
std::vector<Key> random_keys(std::size_t size, std::uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::vector<Key> keys(size);
  for (auto& key : keys) {
    key = static_cast<Key>(rng());
  }
  return keys;
}

template <class Key>
struct cpu_result {
  std::vector<std::uint8_t> bytes;
  double composite_estimate;
};

template <class Key>
cpu_result<Key> make_cpu_result(const std::vector<Key>& keys, std::uint8_t lg_k)
{
  ::datasketches::hll_sketch sketch(lg_k, ::datasketches::HLL_8, /*start_full_size=*/true);
  for (const auto& key : keys) {
    sketch.update(key);
  }

  auto bytes          = sketch.serialize_compact();
  auto composite_blob = bytes;
  composite_blob[flags_offset] =
    static_cast<std::uint8_t>(composite_blob[flags_offset] | ooo_flag_mask);
  auto composite =
    ::datasketches::hll_sketch::deserialize(composite_blob.data(), composite_blob.size());
  return {std::move(bytes), composite.get_estimate()};
}

template <class Register>
void require_register_parity(const std::vector<Register>& registers,
                             const std::vector<std::uint8_t>& cpu_bytes)
{
  REQUIRE(cpu_bytes.size() == register_offset + registers.size());
  for (std::size_t i = 0; i < registers.size(); ++i) {
    CAPTURE(i, registers[i], cpu_bytes[register_offset + i]);
    REQUIRE(static_cast<std::uint8_t>(registers[i]) == cpu_bytes[register_offset + i]);
  }
}

void require_estimate_parity(std::size_t actual, double expected)
{
  const auto truncated  = static_cast<std::size_t>(expected);
  const auto difference = actual > truncated ? actual - truncated : truncated - actual;
  CAPTURE(actual, expected, truncated, difference);
  REQUIRE(difference <= 1);
}

template <class Key>
using device_ref = datasketches::cuda::hll_sketch_ref<Key, ::cuda::thread_scope_device>;

template <class Key>
device_ref<Key> make_ref(thrust::device_vector<typename device_ref<Key>::register_type>& storage)
{
  using register_type = typename device_ref<Key>::register_type;
  auto registers =
    ::cuda::std::span<register_type>{thrust::raw_pointer_cast(storage.data()), storage.size()};
  return device_ref<Key>{::cuda::std::as_writable_bytes(registers)};
}

template <class Ref>
std::size_t estimate_ref(::cuda::stream_ref stream, Ref ref)
{
  thrust::device_vector<std::size_t> result(1);
  estimate_ref_kernel<<<1, 256, 0, stream.get()>>>(ref, thrust::raw_pointer_cast(result.data()));
  REQUIRE(cudaGetLastError() == cudaSuccess);

  std::size_t host_result{};
  REQUIRE(cudaMemcpyAsync(&host_result,
                          thrust::raw_pointer_cast(result.data()),
                          sizeof(host_result),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  return host_result;
}

template <class Ref>
ref_query_result query_ref(::cuda::stream_ref stream, Ref ref, std::uint8_t num_std_dev)
{
  thrust::device_vector<ref_query_result> result(1);
  query_ref_kernel<<<1, 256, 0, stream.get()>>>(
    ref, num_std_dev, thrust::raw_pointer_cast(result.data()));
  REQUIRE(cudaGetLastError() == cudaSuccess);

  ref_query_result host_result{};
  REQUIRE(cudaMemcpyAsync(&host_result,
                          thrust::raw_pointer_cast(result.data()),
                          sizeof(host_result),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  return host_result;
}

}  // namespace

TEST_CASE("hll_sketch_ref exposes the CCCL storage layout", "[hll_ref][api]")
{
  using ref_type  = device_ref<std::uint64_t>;
  using block_ref = datasketches::cuda::hll_sketch_ref<std::uint64_t, ::cuda::thread_scope_block>;
  static_assert(std::is_same_v<typename ref_type::template rebind_scope<::cuda::thread_scope_block>,
                               block_ref>);
  static_assert(ref_type::thread_scope == ::cuda::thread_scope_device);
  static_assert(block_ref::thread_scope == ::cuda::thread_scope_block);
  static_assert(std::is_trivially_copyable_v<ref_type>);

  REQUIRE(ref_type::sketch_alignment() == alignof(ref_type::register_type));
  for (std::uint8_t lg_k = 4; lg_k <= 18; ++lg_k) {
    REQUIRE(ref_type::sketch_bytes(lg_k) ==
            sizeof(ref_type::register_type) * (std::size_t{1} << lg_k));
  }
  std::vector<ref_type::register_type> valid(std::size_t{1} << 8);
  ref_type ref{::cuda::std::as_writable_bytes(
    ::cuda::std::span<ref_type::register_type>{valid.data(), valid.size()})};
  REQUIRE(ref.sketch_bytes() == ref_type::sketch_bytes(8));
  REQUIRE(ref.sketch().data() == reinterpret_cast<::cuda::std::byte*>(valid.data()));
  REQUIRE(ref.get_lg_config_k() == 8);
  REQUIRE(ref.get_target_type() == ::datasketches::HLL_8);
  REQUIRE(ref.num_registers() == 256);

  std::vector<::cuda::std::byte> misaligned(ref_type::sketch_bytes(8) + 1);
  REQUIRE_THROWS_AS((ref_type{::cuda::std::span<::cuda::std::byte>{misaligned.data() + 1,
                                                                   ref_type::sketch_bytes(8)}}),
                    std::invalid_argument);
}

TEST_CASE("device-scope hll_sketch_ref updates caller-owned global storage", "[hll_ref][device]")
{
  constexpr std::uint8_t lg_k = 12;
  auto keys                   = random_keys<std::uint64_t>(100'000, 0xA11CE001ULL);
  const auto cpu              = make_cpu_result(keys, lg_k);

  thrust::device_vector<std::uint64_t> device_keys = keys;
  thrust::device_vector<device_ref<std::uint64_t>::register_type> storage(std::size_t{1} << lg_k);
  auto ref = make_ref<std::uint64_t>(storage);

  ::cuda::stream stream{::cuda::devices[0]};
  clear_ref_kernel<<<1, 256, 0, stream.get()>>>(ref);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  update_ref_kernel<<<128, 256, 0, stream.get()>>>(
    ref, thrust::raw_pointer_cast(device_keys.data()), device_keys.size());
  REQUIRE(cudaGetLastError() == cudaSuccess);

  const auto query = query_ref(stream, ref, /*num_std_dev=*/2);
  require_estimate_parity(query.estimate, cpu.composite_estimate);
  REQUIRE_FALSE(query.empty);
  const auto non_zero = static_cast<double>(
    std::count_if(cpu.bytes.begin() + register_offset, cpu.bytes.end(), [](const auto value) {
      return value != 0;
    }));
  const double lower_rel_error = ::datasketches::HllUtil<>::getRelErr(false, true, lg_k, 2);
  const double upper_rel_error = ::datasketches::HllUtil<>::getRelErr(true, true, lg_k, 2);
  const double expected_device_lower =
    std::max(static_cast<double>(query.estimate) / (1.0 + lower_rel_error), non_zero);
  const double expected_device_upper =
    static_cast<double>(query.estimate) / (1.0 + upper_rel_error);
  REQUIRE(query.lower_bound == Catch::Approx(expected_device_lower).epsilon(1e-15));
  REQUIRE(query.upper_bound == Catch::Approx(expected_device_upper).epsilon(1e-15));
  REQUIRE(query.lg_k == lg_k);
  REQUIRE(query.target_type == static_cast<int>(::datasketches::HLL_8));
  REQUIRE(query.num_registers == storage.size());

  const auto host_estimate = ref.get_estimate(stream);
  require_estimate_parity(host_estimate, cpu.composite_estimate);
  REQUIRE_FALSE(ref.is_empty(stream));
  const double host_lower = ref.get_lower_bound(stream, 2);
  const double host_upper = ref.get_upper_bound(stream, 2);
  const double expected_host_lower =
    std::max(static_cast<double>(host_estimate) / (1.0 + lower_rel_error), non_zero);
  const double expected_host_upper = static_cast<double>(host_estimate) / (1.0 + upper_rel_error);
  REQUIRE(host_lower == Catch::Approx(expected_host_lower).epsilon(1e-15));
  REQUIRE(host_upper == Catch::Approx(expected_host_upper).epsilon(1e-15));
  REQUIRE_THROWS_AS(ref.get_lower_bound(stream, 0), std::invalid_argument);
  REQUIRE_THROWS_AS(ref.get_upper_bound(stream, 4), std::invalid_argument);

  std::vector<device_ref<std::uint64_t>::register_type> registers(storage.size());
  thrust::copy(storage.begin(), storage.end(), registers.begin());
  require_register_parity(registers, cpu.bytes);

  clear_ref_kernel<<<1, 256, 0, stream.get()>>>(ref);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  const auto empty_query = query_ref(stream, ref, /*num_std_dev=*/2);
  REQUIRE(empty_query.estimate == 0);
  REQUIRE(empty_query.lower_bound == 0.0);
  REQUIRE(empty_query.upper_bound == 0.0);
  REQUIRE(empty_query.empty);
  REQUIRE(ref.is_empty(stream));
  thrust::copy(storage.begin(), storage.end(), registers.begin());
  REQUIRE(
    std::all_of(registers.begin(), registers.end(), [](const auto value) { return value == 0; }));
}

TEST_CASE("block-scope hll_sketch_ref operates on block-exclusive shared storage",
          "[hll_ref][block]")
{
  constexpr std::uint8_t lg_k = 8;
  auto keys                   = random_keys<std::uint64_t>(10'000, 0xA11CE002ULL);
  const auto cpu              = make_cpu_result(keys, lg_k);
  thrust::device_vector<std::uint64_t> device_keys = keys;
  thrust::device_vector<std::size_t> result(1);

  ::cuda::stream stream{::cuda::devices[0]};
  using ref_type   = datasketches::cuda::hll_sketch_ref<std::uint64_t, ::cuda::thread_scope_block>;
  const auto bytes = ref_type::sketch_bytes(lg_k);
  block_ref_kernel<<<1, 256, bytes, stream.get()>>>(thrust::raw_pointer_cast(device_keys.data()),
                                                    device_keys.size(),
                                                    bytes,
                                                    thrust::raw_pointer_cast(result.data()));
  REQUIRE(cudaGetLastError() == cudaSuccess);

  std::size_t estimate{};
  REQUIRE(cudaMemcpyAsync(&estimate,
                          thrust::raw_pointer_cast(result.data()),
                          sizeof(estimate),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  require_estimate_parity(estimate, cpu.composite_estimate);
}

TEST_CASE("hll_sketch_ref estimates cover every supported precision", "[hll_ref][estimate]")
{
  ::cuda::stream stream{::cuda::devices[0]};
  for (std::uint8_t lg_k = 4; lg_k <= 18; ++lg_k) {
    const std::uint64_t config_k = std::uint64_t{1} << lg_k;
    const std::uint64_t high_n   = config_k * 32 < 200'000 ? config_k * 32 : 200'000;
    for (std::uint64_t n : {config_k / 4, config_k * 2, high_n}) {
      auto keys = random_keys<std::uint64_t>(n, 0xA11CE100ULL ^ (std::uint64_t{lg_k} << 48) ^ n);
      const auto cpu = make_cpu_result(keys, lg_k);

      std::vector<device_ref<std::uint64_t>::register_type> host_registers(std::size_t{1} << lg_k);
      for (std::size_t i = 0; i < host_registers.size(); ++i) {
        host_registers[i] = cpu.bytes[register_offset + i];
      }

      thrust::device_vector<device_ref<std::uint64_t>::register_type> storage = host_registers;
      CAPTURE(lg_k, n);
      require_estimate_parity(estimate_ref(stream, make_ref<std::uint64_t>(storage)),
                              cpu.composite_estimate);
    }
  }
}

TEST_CASE("hll_sketch_ref cooperatively merges compatible sketches", "[hll_ref][merge]")
{
  constexpr std::uint8_t lg_k         = 12;
  auto first                          = random_keys<std::uint64_t>(50'000, 0xA11CE003ULL);
  auto second                         = random_keys<std::uint64_t>(50'000, 0xA11CE004ULL);
  std::vector<std::uint64_t> combined = first;
  combined.insert(combined.end(), second.begin(), second.end());
  const auto cpu = make_cpu_result(combined, lg_k);

  thrust::device_vector<std::uint64_t> first_device  = first;
  thrust::device_vector<std::uint64_t> second_device = second;
  thrust::device_vector<device_ref<std::uint64_t>::register_type> first_storage(std::size_t{1}
                                                                                << lg_k);
  thrust::device_vector<device_ref<std::uint64_t>::register_type> second_storage(std::size_t{1}
                                                                                 << lg_k);
  auto first_ref         = make_ref<std::uint64_t>(first_storage);
  auto second_device_ref = make_ref<std::uint64_t>(second_storage);
  using block_ref = datasketches::cuda::hll_sketch_ref<std::uint64_t, ::cuda::thread_scope_block>;
  auto second_registers = ::cuda::std::span<block_ref::register_type>{
    thrust::raw_pointer_cast(second_storage.data()), second_storage.size()};
  block_ref second_block_ref{::cuda::std::as_writable_bytes(second_registers)};

  ::cuda::stream stream{::cuda::devices[0]};
  clear_ref_kernel<<<1, 256, 0, stream.get()>>>(first_ref);
  clear_ref_kernel<<<1, 256, 0, stream.get()>>>(second_device_ref);
  update_ref_kernel<<<64, 256, 0, stream.get()>>>(
    first_ref, thrust::raw_pointer_cast(first_device.data()), first_device.size());
  update_ref_kernel<<<64, 256, 0, stream.get()>>>(
    second_device_ref, thrust::raw_pointer_cast(second_device.data()), second_device.size());
  merge_ref_kernel<<<1, 256, 0, stream.get()>>>(first_ref, second_block_ref);
  REQUIRE(cudaGetLastError() == cudaSuccess);

  require_estimate_parity(estimate_ref(stream, first_ref), cpu.composite_estimate);
  std::vector<device_ref<std::uint64_t>::register_type> registers(first_storage.size());
  thrust::copy(first_storage.begin(), first_storage.end(), registers.begin());
  require_register_parity(registers, cpu.bytes);
}

TEST_CASE("hll_sketch ref mutates the owning sketch storage", "[hll_ref][owning]")
{
  constexpr std::uint8_t lg_k = 12;
  auto keys                   = random_keys<std::uint64_t>(100'000, 0xA11CE005ULL);
  const auto cpu              = make_cpu_result(keys, lg_k);
  thrust::device_vector<std::uint64_t> device_keys = keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::hll_sketch<std::uint64_t> sketch(stream, mr, lg_k);
  auto ref = sketch.ref();
  update_ref_kernel<<<128, 256, 0, stream.get()>>>(
    ref, thrust::raw_pointer_cast(device_keys.data()), device_keys.size());
  REQUIRE(cudaGetLastError() == cudaSuccess);

  require_estimate_parity(estimate_ref(stream, ref), cpu.composite_estimate);
  REQUIRE(sketch.get_estimate(stream) == static_cast<double>(ref.get_estimate(stream)));
  REQUIRE(sketch.get_lower_bound(stream, 2) == ref.get_lower_bound(stream, 2));
  REQUIRE(sketch.get_upper_bound(stream, 2) == ref.get_upper_bound(stream, 2));
  REQUIRE(sketch.is_empty(stream) == ref.is_empty(stream));
  const auto bytes = sketch.serialize_compact(stream);
  REQUIRE(bytes.size() == cpu.bytes.size());
  for (std::size_t i = register_offset; i < bytes.size(); ++i) {
    CAPTURE(i);
    REQUIRE(bytes[i] == cpu.bytes[i]);
  }
}

TEST_CASE("moved hll_sketch creates a valid on-demand ref", "[hll_ref][owning][move]")
{
  constexpr std::uint8_t lg_k = 12;
  auto keys                   = random_keys<std::uint64_t>(100'000, 0xA11CE006ULL);
  const auto cpu              = make_cpu_result(keys, lg_k);
  thrust::device_vector<std::uint64_t> device_keys = keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::hll_sketch<std::uint64_t> original(stream, mr, lg_k);
  auto moved = std::move(original);
  auto ref   = moved.ref();

  update_ref_kernel<<<128, 256, 0, stream.get()>>>(
    ref, thrust::raw_pointer_cast(device_keys.data()), device_keys.size());
  REQUIRE(cudaGetLastError() == cudaSuccess);

  require_estimate_parity(ref.get_estimate(stream), cpu.composite_estimate);
  const auto bytes = moved.serialize_compact(stream);
  REQUIRE(bytes.size() == cpu.bytes.size());
  for (std::size_t i = register_offset; i < bytes.size(); ++i) {
    CAPTURE(i);
    REQUIRE(bytes[i] == cpu.bytes[i]);
  }
}
