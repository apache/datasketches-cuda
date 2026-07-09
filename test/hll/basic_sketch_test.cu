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

// End-to-end test for `datasketches::cuda::hll_sketch<uint64_t>` with bulk
// updates. Asserts that the estimate is within the 3-sigma RSE bound for the
// configured `lgK`.

#include <cstdint>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <random>
#include <vector>

#include <thrust/device_vector.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/hll.hpp>

namespace {

constexpr double HLL_RSE_FACTOR = 1.04;  // datasketches HLL_HIP_RSE_FACTOR

double three_sigma_bound(uint8_t lgK)
{
  const double configK = static_cast<double>(1u << lgK);
  return 3.0 * HLL_RSE_FACTOR / std::sqrt(configK);
}

void run_case(uint8_t lgK, uint64_t n, uint64_t seed)
{
  std::vector<uint64_t> host_keys(n);
  std::mt19937_64 rng(seed);
  for (uint64_t i = 0; i < n; ++i)
    host_keys[i] = rng();

  thrust::device_vector<uint64_t> dev_keys = host_keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::hll_sketch<uint64_t> sketch(stream, mr, lgK);
  sketch.update(stream, dev_keys.begin(), dev_keys.end());

  const double est   = sketch.get_estimate(stream);
  const double bound = three_sigma_bound(lgK);
  const double rel   = std::abs(est - static_cast<double>(n)) / static_cast<double>(n);

  INFO("lgK=" << int(lgK) << " n=" << n << " estimate=" << est << " rel_err=" << rel
              << " bound=" << bound);
  REQUIRE(rel < bound);
}

}  // namespace

TEST_CASE("HLL_8 estimate within 3-sigma RSE", "[hll_sketch][basic]")
{
  // Pairs (lgK, n) chosen so that n is well above the LIST/SET threshold and
  // landings span the cubic and asymptote branches of the Composite blender.
  run_case(8, 10'000, 0xDEADBEEFC0FFEE08ULL);
  run_case(12, 100'000, 0xDEADBEEFC0FFEE12ULL);
  run_case(16, 500'000, 0xDEADBEEFC0FFEE16ULL);
  run_case(18, 1'000'000, 0xDEADBEEFC0FFEE18ULL);
}

TEST_CASE("HLL_8 throws on non-HLL_8 target", "[hll_sketch][basic]")
{
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  REQUIRE_THROWS_AS(
    (datasketches::cuda::hll_sketch<uint64_t>(stream, mr, 12, datasketches::cuda::HLL_4)),
    std::invalid_argument);
  REQUIRE_THROWS_AS(
    (datasketches::cuda::hll_sketch<uint64_t>(stream, mr, 12, datasketches::cuda::HLL_6)),
    std::invalid_argument);
}

TEST_CASE("HLL_8 is_empty after construction", "[hll_sketch][basic]")
{
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::hll_sketch<uint64_t> sketch(stream, mr, 12);
  REQUIRE(sketch.is_empty(stream));
  REQUIRE(sketch.get_lg_config_k() == 12);
  REQUIRE(sketch.get_target_type() == datasketches::cuda::HLL_8);
  REQUIRE(sketch.num_registers() == 4096u);
}

TEST_CASE("HLL_8 lower_bound <= estimate <= upper_bound", "[hll_sketch][basic]")
{
  std::vector<uint64_t> host_keys(100'000);
  std::mt19937_64 rng(0xABCD0001ULL);
  for (auto& k : host_keys)
    k = rng();
  thrust::device_vector<uint64_t> dev_keys = host_keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::hll_sketch<uint64_t> sketch(stream, mr, 12);
  sketch.update(stream, dev_keys.begin(), dev_keys.end());

  const double lb = sketch.get_lower_bound(stream, 2);
  const double e  = sketch.get_estimate(stream);
  const double ub = sketch.get_upper_bound(stream, 2);
  REQUIRE(lb <= e);
  REQUIRE(e <= ub);
}

TEST_CASE("HLL_8 ctor uses caller-supplied stream and memory resource",
          "[hll_sketch][basic][stream]")
{
  using Catch::Approx;

  constexpr uint8_t lgK = 12;
  constexpr uint64_t n  = 100'000;

  std::vector<uint64_t> host_keys(n);
  std::mt19937_64 rng(0xABCD0002ULL);
  for (auto& k : host_keys)
    k = rng();
  thrust::device_vector<uint64_t> dev_keys = host_keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::hll_sketch<uint64_t> a(stream, mr, lgK);
  datasketches::cuda::hll_sketch<uint64_t> b(stream, mr, lgK);

  a.update(stream, dev_keys.begin(), dev_keys.end());
  b.update(stream, dev_keys.begin(), dev_keys.end());

  REQUIRE(a.get_estimate(stream) == Approx(b.get_estimate(stream)).epsilon(1e-12));
}

TEST_CASE("HLL_8 deserialize uses caller-supplied stream and memory resource",
          "[hll_sketch][basic][stream]")
{
  using Catch::Approx;

  constexpr uint8_t lgK = 12;
  constexpr uint64_t n  = 100'000;

  std::vector<uint64_t> host_keys(n);
  std::mt19937_64 rng(0xABCD0003ULL);
  for (auto& k : host_keys)
    k = rng();
  thrust::device_vector<uint64_t> dev_keys = host_keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::hll_sketch<uint64_t> src(stream, mr, lgK);
  src.update(stream, dev_keys.begin(), dev_keys.end());
  const auto bytes = src.serialize_compact(stream);

  auto dst = datasketches::cuda::hll_sketch<uint64_t>::deserialize(
    stream, ::cuda::std::span<const std::uint8_t>{bytes.data(), bytes.size()}, mr);

  REQUIRE(dst.get_estimate(stream) == Approx(src.get_estimate(stream)).epsilon(1e-12));
}
