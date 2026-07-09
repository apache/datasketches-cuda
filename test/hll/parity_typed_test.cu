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

// Byte-level CPU/GPU parity test for every supported Key type.
//
// For each type, the same sequence of values is fed to:
//   - CPU: datasketches::hll_sketch (start_full_size=true, HLL_8)
//   - GPU: datasketches::cuda::hll_sketch<Key>
//
// After updates, both are serialized compact and compared:
//   - Register bytes (offset 40+) must be byte-for-byte identical.
//   - kxq0, kxq1, numAtCurMin (deterministic from registers) must match.
//
// Verifies that normalizing_hasher<Key> replicates the normalization in each
// datasketches-cpp hll_sketch_alloc::update(...) overload on the device.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <limits>
#include <random>
#include <vector>

#include <thrust/device_vector.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <hll.hpp>

#include <datasketches/cuda/hll.hpp>

namespace {

constexpr std::size_t REG_OFF        = 40;
constexpr std::size_t KXQ0_OFF       = 16;
constexpr std::size_t KXQ1_OFF       = 24;
constexpr std::size_t NUM_AT_MIN_OFF = 32;

template <class T>
void compare_cpu_gpu(const std::vector<T>& keys, uint8_t lgK)
{
  // CPU sketch
  ::datasketches::hll_sketch cpu(lgK, ::datasketches::HLL_8, /*start_full_size=*/true);
  for (const T& k : keys)
    cpu.update(k);
  auto cpu_bytes = cpu.serialize_compact();

  // GPU sketch
  thrust::device_vector<T> dev_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::hll_sketch<T> gpu(stream, mr, lgK);
  gpu.update(stream, dev_keys.begin(), dev_keys.end());
  auto gpu_bytes = gpu.serialize_compact(stream);

  REQUIRE(cpu_bytes.size() == gpu_bytes.size());
  REQUIRE(cpu_bytes.size() == REG_OFF + (std::size_t{1} << lgK));

  // 1. Register array byte-for-byte equal.
  for (std::size_t i = REG_OFF; i < cpu_bytes.size(); ++i) {
    if (cpu_bytes[i] != gpu_bytes[i]) {
      INFO("lgK=" << int(lgK) << " register slot " << (i - REG_OFF) << " cpu=" << int(cpu_bytes[i])
                  << " gpu=" << int(gpu_bytes[i]));
      REQUIRE(cpu_bytes[i] == gpu_bytes[i]);
    }
  }

  // 2. kxq0/kxq1/numAtCurMin.
  double cpu_kxq0{}, cpu_kxq1{}, gpu_kxq0{}, gpu_kxq1{};
  std::memcpy(&cpu_kxq0, cpu_bytes.data() + KXQ0_OFF, sizeof(double));
  std::memcpy(&cpu_kxq1, cpu_bytes.data() + KXQ1_OFF, sizeof(double));
  std::memcpy(&gpu_kxq0, gpu_bytes.data() + KXQ0_OFF, sizeof(double));
  std::memcpy(&gpu_kxq1, gpu_bytes.data() + KXQ1_OFF, sizeof(double));
  std::uint32_t cpu_num{}, gpu_num{};
  std::memcpy(&cpu_num, cpu_bytes.data() + NUM_AT_MIN_OFF, sizeof(std::uint32_t));
  std::memcpy(&gpu_num, gpu_bytes.data() + NUM_AT_MIN_OFF, sizeof(std::uint32_t));
  REQUIRE(cpu_kxq0 == Catch::Approx(gpu_kxq0).epsilon(1e-12));
  REQUIRE(cpu_kxq1 == Catch::Approx(gpu_kxq1).epsilon(1e-12));
  REQUIRE(cpu_num == gpu_num);
}

// ---- Random key generators ----

template <class T>
std::vector<T> random_integers(std::size_t n, uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::vector<T> keys(n);
  for (auto& k : keys)
    k = static_cast<T>(rng());
  return keys;
}

// All representable values of T (useful for 8-bit and 16-bit types).
template <class T>
std::vector<T> all_values()
{
  std::vector<T> v;
  T lo = std::numeric_limits<T>::min();
  T hi = std::numeric_limits<T>::max();
  for (T x = lo;; ++x) {
    v.push_back(x);
    if (x == hi) break;
  }
  return v;
}

// Edge cases for floating-point types.
template <class Float>
std::vector<Float> float_edge_cases()
{
  auto bits_as = [](auto bits) {
    Float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  };
  std::vector<Float> v;

  // +0.0, -0.0
  v.push_back(static_cast<Float>(0.0));
  v.push_back(static_cast<Float>(-0.0));

  // +inf, -inf
  v.push_back(std::numeric_limits<Float>::infinity());
  v.push_back(-std::numeric_limits<Float>::infinity());

  // Several distinct NaN payloads (positive quiet, negative quiet, with payload).
  if constexpr (sizeof(Float) == 4) {
    v.push_back(bits_as(uint32_t{0x7fc00000u}));  // positive quiet NaN
    v.push_back(bits_as(uint32_t{0xffc00000u}));  // negative quiet NaN
    v.push_back(bits_as(uint32_t{0x7fc00042u}));  // quiet NaN with payload
    v.push_back(bits_as(uint32_t{0x7f800001u}));  // signaling NaN
  } else {
    v.push_back(bits_as(uint64_t{0x7ff8000000000000ULL}));  // positive quiet NaN
    v.push_back(bits_as(uint64_t{0xfff8000000000000ULL}));  // negative quiet NaN
    v.push_back(bits_as(uint64_t{0x7ff8000000000042ULL}));  // quiet NaN with payload
    v.push_back(bits_as(uint64_t{0x7ff0000000000001ULL}));  // signaling NaN
  }

  // Subnormals
  v.push_back(std::numeric_limits<Float>::denorm_min());
  v.push_back(-std::numeric_limits<Float>::denorm_min());

  return v;
}

}  // namespace

// ============================================================
// Integral types
// ============================================================

TEST_CASE("parity HLL_8 uint64_t", "[parity_typed]")
{
  compare_cpu_gpu(random_integers<uint64_t>(10'000, 0xABC1u), 12);
}

TEST_CASE("parity HLL_8 int64_t", "[parity_typed]")
{
  auto keys = random_integers<int64_t>(10'000, 0xABC2u);
  // Ensure INT64_MIN and INT64_MAX appear.
  keys.push_back(std::numeric_limits<int64_t>::min());
  keys.push_back(std::numeric_limits<int64_t>::max());
  compare_cpu_gpu(keys, 12);
}

TEST_CASE("parity HLL_8 uint32_t", "[parity_typed]")
{
  auto keys = random_integers<uint32_t>(10'000, 0xABC3u);
  keys.push_back(0u);
  keys.push_back(std::numeric_limits<uint32_t>::max());  // 0xFFFFFFFF → sign-ext to -1
  keys.push_back(0x80000000u);                           // maps to INT32_MIN after reinterpret
  compare_cpu_gpu(keys, 12);
}

TEST_CASE("parity HLL_8 int32_t", "[parity_typed]")
{
  auto keys = random_integers<int32_t>(10'000, 0xABC4u);
  keys.push_back(std::numeric_limits<int32_t>::min());
  keys.push_back(std::numeric_limits<int32_t>::max());
  keys.push_back(-1);
  compare_cpu_gpu(keys, 12);
}

TEST_CASE("parity HLL_8 uint16_t — full range", "[parity_typed]")
{
  // All 65536 values: complete coverage of sign-reinterpret path.
  compare_cpu_gpu(all_values<uint16_t>(), 8);
}

TEST_CASE("parity HLL_8 int16_t — full range", "[parity_typed]")
{
  compare_cpu_gpu(all_values<int16_t>(), 8);
}

TEST_CASE("parity HLL_8 uint8_t — full range repeated", "[parity_typed]")
{
  // All 256 values repeated 40 times so the sketch has enough distinct inputs
  // to exercise a meaningful register distribution.
  auto base = all_values<uint8_t>();
  std::vector<uint8_t> keys;
  keys.reserve(base.size() * 40);
  for (int i = 0; i < 40; ++i)
    keys.insert(keys.end(), base.begin(), base.end());
  compare_cpu_gpu(keys, 8);
}

TEST_CASE("parity HLL_8 int8_t — full range repeated", "[parity_typed]")
{
  auto base = all_values<int8_t>();
  std::vector<int8_t> keys;
  keys.reserve(base.size() * 40);
  for (int i = 0; i < 40; ++i)
    keys.insert(keys.end(), base.begin(), base.end());
  compare_cpu_gpu(keys, 8);
}

// ============================================================
// Floating-point types
// ============================================================

TEST_CASE("parity HLL_8 double — random + edge cases", "[parity_typed]")
{
  // Random doubles from uniform distribution + critical edge cases.
  std::mt19937_64 rng(0xABC9u);
  std::uniform_real_distribution<double> dist(-1e15, 1e15);
  std::vector<double> keys;
  for (int i = 0; i < 9'000; ++i)
    keys.push_back(dist(rng));

  auto edges = float_edge_cases<double>();
  keys.insert(keys.end(), edges.begin(), edges.end());

  compare_cpu_gpu(keys, 12);
}

TEST_CASE("parity HLL_8 float — random + edge cases", "[parity_typed]")
{
  std::mt19937_64 rng(0xABCAu);
  std::uniform_real_distribution<float> dist(-1e10f, 1e10f);
  std::vector<float> keys;
  for (int i = 0; i < 9'000; ++i)
    keys.push_back(dist(rng));

  auto edges = float_edge_cases<float>();
  keys.insert(keys.end(), edges.begin(), edges.end());

  compare_cpu_gpu(keys, 12);
}
