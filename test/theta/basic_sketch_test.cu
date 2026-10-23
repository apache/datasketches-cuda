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
#include <cmath>
#include <cstdint>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>
#include <limits>
#include <new>
#include <numeric>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/theta.hpp>

#include "cpu_reference.hpp"

namespace {

struct allocation_failure {
  std::size_t allocations = 0;
  std::size_t fail_at     = std::numeric_limits<std::size_t>::max();

  void check()
  {
    if (allocations++ == fail_at) throw std::bad_alloc{};
  }
};

struct failing_resource {
  ::cuda::device_memory_pool_ref upstream;
  allocation_failure* failure;

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    failure->check();
    return upstream.allocate_sync(bytes, alignment);
  }

  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    failure->check();
    return upstream.allocate(stream, bytes, alignment);
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    upstream.deallocate_sync(ptr, bytes, alignment);
  }

  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    upstream.deallocate(stream, ptr, bytes, alignment);
  }

  bool operator==(const failing_resource& other) const noexcept
  {
    return upstream == other.upstream && failure == other.failure;
  }

  bool operator!=(const failing_resource& other) const noexcept { return !(*this == other); }

  friend void get_property(const failing_resource&, ::cuda::mr::device_accessible) noexcept {}
};

thrust::device_vector<std::uint64_t> make_keys(std::size_t count, std::uint64_t first)
{
  thrust::device_vector<std::uint64_t> keys(count);
  thrust::sequence(thrust::device, keys.begin(), keys.end(), first);
  return keys;
}

using theta_test::cpu_image;

std::vector<std::uint8_t> cpu_image(const thrust::device_vector<std::uint64_t>& keys,
                                    std::uint8_t lg_k,
                                    std::uint64_t seed,
                                    float p = 1.0F)
{
  std::vector<std::uint64_t> host(keys.size());
  thrust::copy(keys.begin(), keys.end(), host.begin());
  return theta_test::cpu_image(host, lg_k, seed, p);
}

}  // namespace

TEST_CASE("Theta starts empty and exact", "[theta][basic]")
{
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> sketch(stream, mr, 12);

  REQUIRE(sketch.is_empty());
  REQUIRE(sketch.is_ordered());
  REQUIRE_FALSE(sketch.is_estimation_mode());
  REQUIRE(sketch.get_lg_k() == 12);
  REQUIRE(sketch.get_num_retained() == 0);
  REQUIRE(sketch.get_estimate() == 0.0);
  REQUIRE(sketch.get_theta() == 1.0);

  REQUIRE(sketch.serialize_compact(stream) == cpu_image(std::vector<std::uint64_t>{}, 12, 9001));
}

TEST_CASE("Theta failed first chunk preserves the empty sketch", "[theta][exception][resource]")
{
  constexpr std::uint8_t lg_k  = 8;
  constexpr std::uint64_t seed = 9001;
  constexpr float p            = 0.125F;
  const auto keys              = make_keys(1000, 0xa110cULL);

  ::cuda::stream allocation_stream{::cuda::devices[0]};
  ::cuda::stream operation_stream{::cuda::devices[0]};
  allocation_failure failure;
  failing_resource mr{::cuda::device_default_memory_pool(::cuda::devices[0]), &failure};
  datasketches::cuda::theta_sketch<std::uint64_t, failing_resource> sketch(
    allocation_stream, mr, lg_k, seed, p);

  SECTION("first allocation") { failure.fail_at = 0; }
  SECTION("final allocation after screening and sorting")
  {
    // Learn the successful path's allocation count instead of depending on how
    // many temporary allocations this CUB version makes.
    failure.allocations = 0;
    sketch.update(operation_stream, keys.begin(), keys.end());
    REQUIRE(failure.allocations > 1);
    const auto final_allocation = failure.allocations - 1;
    sketch.reset(operation_stream);
    failure.fail_at = final_allocation;
  }
  const auto before = sketch.serialize_compact(operation_stream);

  failure.allocations = 0;
  // CUB translates allocator exceptions into CUDA errors.
  REQUIRE_THROWS(sketch.update(operation_stream, keys.begin(), keys.end()));
  REQUIRE(failure.allocations == failure.fail_at + 1);
  failure.fail_at = std::numeric_limits<std::size_t>::max();
  REQUIRE(sketch.is_empty());
  REQUIRE(sketch.serialize_compact(operation_stream) == before);

  sketch.update(operation_stream, keys.begin(), keys.end());
  REQUIRE(sketch.serialize_compact(operation_stream) == cpu_image(keys, lg_k, seed, p));
}

TEST_CASE("Theta compact v3 bytes match trimmed CPU in estimation mode",
          "[theta][parity][serialization]")
{
  constexpr std::uint8_t lg_k  = 10;
  constexpr std::uint64_t seed = 123456789;
  auto keys                    = make_keys(100000, 0xabcdefULL);

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k, seed);
  gpu.update(stream, keys.begin(), keys.end());

  REQUIRE(gpu.is_estimation_mode());
  REQUIRE(gpu.get_num_retained() == (std::size_t{1} << lg_k));
  REQUIRE(gpu.serialize_compact(stream) == cpu_image(keys, lg_k, seed));

  const double estimate = gpu.get_estimate();
  const double lower    = gpu.get_lower_bound(2);
  const double upper    = gpu.get_upper_bound(2);
  REQUIRE(lower <= estimate);
  REQUIRE(estimate <= upper);
}

TEST_CASE("Theta p-sampling and custom seed match CPU", "[theta][parity][sampling]")
{
  constexpr std::uint8_t lg_k  = 12;
  constexpr std::uint64_t seed = 0xdecafbadULL;
  constexpr float p            = 0.125F;
  auto keys                    = make_keys(10000, 0xf00dULL);

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k, seed, p);
  gpu.update(stream, keys.begin(), keys.end());

  const auto bytes = gpu.serialize_compact(stream);
  REQUIRE(bytes == cpu_image(keys, lg_k, seed, p));

  gpu.reset(stream);
  REQUIRE(gpu.is_empty());
  REQUIRE(gpu.get_num_retained() == 0);
  gpu.update(stream, keys.begin(), keys.end());
  REQUIRE(gpu.serialize_compact(stream) == bytes);
}

TEST_CASE("Theta imports and updates CPU compact v3 images", "[theta][serialization][parity]")
{
  constexpr std::uint8_t lg_k  = 8;
  constexpr std::uint64_t seed = 0xc0ffeeULL;
  using sketch_type            = datasketches::cuda::theta_sketch<std::uint64_t>;
  std::vector<std::uint64_t> keys;
  float p = 1.0F;
  SECTION("empty") {}
  SECTION("single entry") { keys = {42}; }
  SECTION("exact") { keys.resize(100); }
  SECTION("estimation") { keys.resize(10000); }
  SECTION("sampled with zero retained entries")
  {
    keys = {42};
    p    = 0x1p-63F;
  }

  if (keys.size() > 1) std::iota(keys.begin(), keys.end(), std::uint64_t{1});
  const auto bytes = cpu_image(keys, lg_k, seed, p);
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr     = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  auto sketch = sketch_type::deserialize(
    stream, ::cuda::std::span<const std::uint8_t>{bytes.data(), bytes.size()}, mr, lg_k, seed);
  REQUIRE(sketch.serialize_compact(stream) == bytes);
  REQUIRE(sketch.is_empty() == keys.empty());

  const std::vector<std::uint64_t> additional{42, 43, 44};
  thrust::device_vector<std::uint64_t> device = additional;
  sketch.update(stream, device.begin(), device.end());
  keys.insert(keys.end(), additional.begin(), additional.end());
  REQUIRE(sketch.serialize_compact(stream) == cpu_image(keys, lg_k, seed, p));
}

TEST_CASE("Theta validates constructor and compact image", "[theta][validation]")
{
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  REQUIRE_THROWS_AS((datasketches::cuda::theta_sketch<std::uint64_t>(stream, mr, 4)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((datasketches::cuda::theta_sketch<std::uint64_t>(stream, mr, 12, 9001, 0.0F)),
                    std::invalid_argument);
  // Below 2^-63 the hash threshold rounds to zero.
  REQUIRE_THROWS_AS((datasketches::cuda::theta_sketch<std::uint64_t>(
                      stream, mr, 12, 9001, std::nextafter(0x1p-63F, 0.0F))),
                    std::invalid_argument);

  std::vector<std::uint8_t> invalid(8, 0);
  invalid[0] = 1;
  invalid[1] = 4;
  invalid[2] = 3;
  REQUIRE_THROWS_AS(
    datasketches::cuda::theta_sketch<std::uint64_t>::deserialize(
      stream, ::cuda::std::span<const std::uint8_t>{invalid.data(), invalid.size()}, mr),
    std::invalid_argument);

  std::vector<std::uint8_t> truncated(8, 0);
  truncated[0] = 3;
  truncated[1] = 3;
  truncated[2] = 3;
  truncated[5] = (1U << 1) | (1U << 3) | (1U << 4);
  datasketches::cuda::theta_sketch<std::uint64_t> seed_source(stream, mr);
  const auto seed_hash = seed_source.get_seed_hash();
  truncated[6]         = static_cast<std::uint8_t>(seed_hash);
  truncated[7]         = static_cast<std::uint8_t>(seed_hash >> 8);
  REQUIRE_THROWS_AS(
    datasketches::cuda::theta_sketch<std::uint64_t>::deserialize(
      stream, ::cuda::std::span<const std::uint8_t>{truncated.data(), truncated.size()}, mr),
    std::invalid_argument);

  auto trailing = cpu_image(std::vector<std::uint64_t>{42}, 12, 9001);
  trailing.push_back(0);
  REQUIRE_THROWS_AS(
    datasketches::cuda::theta_sketch<std::uint64_t>::deserialize(
      stream, ::cuda::std::span<const std::uint8_t>{trailing.data(), trailing.size()}, mr),
    std::invalid_argument);
}

TEST_CASE("Theta batch splitting does not change the result", "[theta][parity][chunking]")
{
  // The same keys fed as one call and as several must produce identical images,
  // whether the split is the caller's or update()'s own.
  constexpr std::uint8_t lg_k  = 11;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(3'000'000, 0xfeedfaceULL);

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  datasketches::cuda::theta_sketch<std::uint64_t> single(stream, mr, lg_k, seed);
  single.update(stream, keys.begin(), keys.end());

  datasketches::cuda::theta_sketch<std::uint64_t> split(stream, mr, lg_k, seed);
  const std::size_t batch = 700'000;
  for (std::size_t offset = 0; offset < keys.size(); offset += batch) {
    const auto end = std::min(offset + batch, keys.size());
    split.update(stream, keys.begin() + offset, keys.begin() + end);
  }

  REQUIRE(single.serialize_compact(stream) == split.serialize_compact(stream));
  REQUIRE(single.serialize_compact(stream) == cpu_image(keys, lg_k, seed));
}

TEST_CASE("Theta grouped duplicates followed by distinct keys match CPU",
          "[theta][parity][chunking][budget]")
{
  // An abrupt transition from grouped duplicates to distinct keys must remain
  // correct regardless of how chunks are sized or retried.
  using sketch_type                      = datasketches::cuda::theta_sketch<std::uint64_t>;
  constexpr std::uint8_t lg_k            = 12;
  constexpr std::uint64_t seed           = 9001;
  constexpr std::size_t run              = 2048;
  constexpr std::size_t grouped_distinct = 4096;
  constexpr std::size_t grouped_keys     = run * grouped_distinct;  // 8M keys
  constexpr std::size_t distinct_keys    = std::size_t{1} << 24;    // 16M keys
  std::vector<std::uint64_t> keys(grouped_keys + distinct_keys);
  for (std::size_t i = 0; i < grouped_keys; ++i)
    keys[i] = 0x1000'0000'0000'0000ULL + i / run;
  for (std::size_t i = 0; i < distinct_keys; ++i)
    keys[grouped_keys + i] = 0x2000'0000'0000'0000ULL + i;

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  sketch_type gpu(stream, mr, lg_k, seed, 1.0F);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.serialize_compact(stream) == cpu_image(keys, lg_k, seed));
}

TEST_CASE("Theta screen overflow retries preserve the result", "[theta][parity][chunking][budget]")
{
  // Force bounded screen writes, then compare retrying updates with the CPU,
  // including when the caller also splits the input.
  using impl_type              = datasketches::cuda::detail::theta::sketch_impl<std::uint64_t>;
  constexpr std::uint8_t lg_k  = 8;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(20'000, 0x0ff10adULL);
  keys.resize(25'000);
  thrust::copy_n(thrust::device, keys.begin(), 5'000, keys.begin() + 20'000);  // duplicates
  thrust::shuffle(thrust::device, keys.begin(), keys.end(), thrust::default_random_engine{0x5eed});

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  impl_type tiny(stream, mr, lg_k, seed, 1.0F);
  tiny.survivor_budget_ = 64;
  {
    const auto result = tiny.screen(stream, keys.begin(), keys.end() - keys.begin());
    REQUIRE(result.data.size() == 64);
    REQUIRE(result.size > result.data.size());
  }
  tiny.update(stream, keys.begin(), keys.end());

  const auto bytes = tiny.serialize_compact(stream);
  REQUIRE(bytes == cpu_image(keys, lg_k, seed));

  // The same holds when the keys arrive in several calls, so the retry path
  // composes with the caller's own batching.
  impl_type batched(stream, mr, lg_k, seed, 1.0F);
  batched.survivor_budget_ = 64;
  const std::size_t batch  = 7'001;
  for (std::size_t offset = 0; offset < keys.size(); offset += batch) {
    const auto end = std::min(offset + batch, keys.size());
    batched.update(stream, keys.begin() + offset, keys.begin() + end);
  }
  REQUIRE(batched.serialize_compact(stream) == bytes);
}

TEST_CASE("Theta batch far above the budget with few distinct keys matches CPU",
          "[theta][parity][chunking][budget]")
{
  // Fewer than k distinct keys keeps theta at its maximum even for a large
  // input whose duplicates are scattered.
  using sketch_type              = datasketches::cuda::theta_sketch<std::uint64_t>;
  constexpr std::uint8_t lg_k    = 12;
  constexpr std::uint64_t seed   = 9001;
  constexpr std::size_t count    = std::size_t{3} << 22;
  constexpr std::size_t distinct = 3'000;
  constexpr std::uint64_t first  = 0xd15c0ULL;
  std::vector<std::uint64_t> keys(count);
  for (std::size_t i = 0; i < count; ++i)
    keys[i] = first + (i * 7919) % distinct;

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  sketch_type gpu(stream, mr, lg_k, seed, 1.0F);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.get_estimate() == static_cast<double>(distinct));
  REQUIRE(gpu.serialize_compact(stream) == cpu_image(keys, lg_k, seed));
}

TEST_CASE("Theta large lg_k matches CPU", "[theta][parity][chunking][budget]")
{
  using sketch_type            = datasketches::cuda::theta_sketch<std::uint64_t>;
  constexpr std::uint8_t lg_k  = 20;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(5'000'000, 0xb16b00b5ULL);

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  sketch_type gpu(stream, mr, lg_k, seed, 1.0F);
  gpu.update(stream, keys.begin(), keys.end());

  REQUIRE(gpu.serialize_compact(stream) == cpu_image(keys, lg_k, seed));
}
