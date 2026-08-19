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

#include <cstdint>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>
#include <random>
#include <vector>

#include <thrust/device_vector.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/theta.hpp>

#include "cpu_reference.hpp"

namespace {

std::vector<std::uint64_t> make_keys(std::size_t count, std::uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::vector<std::uint64_t> keys(count);
  for (auto& key : keys)
    key = rng();
  return keys;
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

  const auto bytes = sketch.serialize_compact(stream);
  REQUIRE(bytes == theta_test::cpu_image(std::vector<std::uint64_t>{}, 12, 9001));
  const auto cpu = theta_test::cpu_metadata(bytes);
  REQUIRE(cpu.empty);
  REQUIRE(cpu.ordered);
}

TEST_CASE("Theta compact v3 bytes match CPU in exact mode", "[theta][parity][serialization]")
{
  constexpr std::uint8_t lg_k  = 12;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(1000, 0x12345678ULL);
  keys.insert(keys.end(), keys.begin(), keys.begin() + 200);

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k, seed);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));
  REQUIRE(gpu.get_num_retained() == 1000);
  REQUIRE(gpu.get_estimate() == 1000.0);
}

TEST_CASE("Theta compact v3 bytes match trimmed CPU in estimation mode",
          "[theta][parity][serialization]")
{
  constexpr std::uint8_t lg_k                      = 10;
  constexpr std::uint64_t seed                     = 123456789;
  auto keys                                        = make_keys(100000, 0xabcdefULL);
  thrust::device_vector<std::uint64_t> device_keys = keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k, seed);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.is_estimation_mode());
  REQUIRE(gpu.get_num_retained() == (std::size_t{1} << lg_k));
  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));

  const double estimate = gpu.get_estimate();
  const double lower    = gpu.get_lower_bound(2);
  const double upper    = gpu.get_upper_bound(2);
  REQUIRE(lower <= estimate);
  REQUIRE(estimate <= upper);
}

TEST_CASE("Theta incremental batches match a single CPU update sketch",
          "[theta][parity][incremental]")
{
  constexpr std::uint8_t lg_k                      = 10;
  auto keys                                        = make_keys(50000, 0x31415926ULL);
  thrust::device_vector<std::uint64_t> device_keys = keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k);
  gpu.update(stream, device_keys.begin(), device_keys.begin() + 7777);
  gpu.update(stream, device_keys.begin() + 7777, device_keys.begin() + 23456);
  gpu.update(stream, device_keys.begin() + 23456, device_keys.end());

  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, 9001));
}

TEST_CASE("Theta p-sampling and custom seed match CPU", "[theta][parity][sampling]")
{
  constexpr std::uint8_t lg_k                      = 12;
  constexpr std::uint64_t seed                     = 0xdecafbadULL;
  constexpr float p                                = 0.125F;
  auto keys                                        = make_keys(10000, 0xf00dULL);
  thrust::device_vector<std::uint64_t> device_keys = keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k, seed, p);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed, p));
}

TEST_CASE("Theta compact v3 round trips through CPU and GPU", "[theta][serialization]")
{
  constexpr std::uint8_t lg_k                      = 10;
  auto keys                                        = make_keys(10000, 0xfeedULL);
  thrust::device_vector<std::uint64_t> device_keys = keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> source(stream, mr, lg_k);
  source.update(stream, device_keys.begin(), device_keys.end());
  const auto bytes = source.serialize_compact(stream);

  const auto cpu = theta_test::cpu_metadata(bytes);
  REQUIRE(cpu.retained == source.get_num_retained());
  REQUIRE(cpu.theta == source.get_theta64());

  auto restored = datasketches::cuda::theta_sketch<std::uint64_t>::deserialize(
    stream, ::cuda::std::span<const std::uint8_t>{bytes.data(), bytes.size()}, mr, lg_k);
  REQUIRE(restored.serialize_compact(stream) == bytes);
  REQUIRE(restored.get_estimate() == Catch::Approx(source.get_estimate()));
}

TEST_CASE("Theta validates constructor and compact image", "[theta][validation]")
{
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  REQUIRE_THROWS_AS((datasketches::cuda::theta_sketch<std::uint64_t>(stream, mr, 4)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((datasketches::cuda::theta_sketch<std::uint64_t>(stream, mr, 12, 9001, 0.0F)),
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
}

TEST_CASE("Theta multi-chunk update matches a single CPU sketch", "[theta][parity][chunking]")
{
  // Large enough that update() splits the batch internally: the sketch enters
  // with theta at its maximum, so the first chunk is bounded and later chunks
  // run against a tightened theta. Parity with the CPU sketch must not depend
  // on how the batch happens to be split.
  constexpr std::uint8_t lg_k  = 12;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(5'000'000, 0xc0ffeeULL);

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k, seed);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.is_estimation_mode());
  REQUIRE(gpu.get_num_retained() == (std::size_t{1} << lg_k));
  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));
}

TEST_CASE("Theta batch splitting does not change the result", "[theta][chunking]")
{
  // The same keys fed as one call and as several must produce identical images,
  // whether the split is the caller's or update()'s own.
  constexpr std::uint8_t lg_k  = 11;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(3'000'000, 0xfeedfaceULL);

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  datasketches::cuda::theta_sketch<std::uint64_t> single(stream, mr, lg_k, seed);
  single.update(stream, device_keys.begin(), device_keys.end());

  datasketches::cuda::theta_sketch<std::uint64_t> split(stream, mr, lg_k, seed);
  const std::size_t batch = 700'000;
  for (std::size_t offset = 0; offset < device_keys.size(); offset += batch) {
    const auto end = std::min(offset + batch, device_keys.size());
    split.update(stream, device_keys.begin() + offset, device_keys.begin() + end);
  }

  REQUIRE(single.serialize_compact(stream) == split.serialize_compact(stream));
  REQUIRE(single.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));
}
