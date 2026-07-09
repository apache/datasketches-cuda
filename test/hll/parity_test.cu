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

// End-to-end CPU/GPU parity gate. Feeds the same N keys + lgK to both
// `datasketches::hll_sketch(lgK, HLL_8, start_full_size=true)` and
// `datasketches::cuda::hll_sketch<uint64_t>(stream, mr, lgK, HLL_8)`. Compares:
//   - Register bytes (offset 40+) byte-for-byte (proves hash + bit-slicing parity).
//   - kxq0/kxq1/numAtCurMin in the preamble (deterministic from registers).
//   - Estimates within the 3-sigma RSE bound.
//
// Failure of register-byte equality pinpoints divergence in (a) hash output
// endianness (h1/h2 swap), (b) bit-slicing width, or (c) register storage order.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>
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

double three_sigma_bound(uint8_t lgK)
{
  return 3.0 * 1.04 / std::sqrt(static_cast<double>(1u << lgK));
}

void run(uint8_t lgK, uint64_t n, uint64_t seed)
{
  std::vector<uint64_t> keys(n);
  std::mt19937_64 rng(seed);
  for (auto& k : keys)
    k = rng();

  // CPU side: start_full_size=true so the sketch is in HLL mode from the start.
  ::datasketches::hll_sketch cpu(lgK, ::datasketches::HLL_8, /*start_full_size=*/true);
  for (uint64_t k : keys)
    cpu.update(k);
  auto cpu_bytes = cpu.serialize_compact();

  // GPU side.
  thrust::device_vector<uint64_t> dev_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::hll_sketch<uint64_t> gpu(stream, mr, lgK);
  gpu.update(stream, dev_keys.begin(), dev_keys.end());
  auto gpu_bytes = gpu.serialize_compact(stream);

  REQUIRE(cpu_bytes.size() == gpu_bytes.size());
  REQUIRE(cpu_bytes.size() == REG_OFF + (1u << lgK));

  // 1. Register array byte-for-byte equal.
  for (std::size_t i = REG_OFF; i < cpu_bytes.size(); ++i) {
    if (cpu_bytes[i] != gpu_bytes[i]) {
      INFO("lgK=" << int(lgK) << " n=" << n << " register slot " << (i - REG_OFF)
                  << " cpu=" << int(cpu_bytes[i]) << " gpu=" << int(gpu_bytes[i]));
      REQUIRE(cpu_bytes[i] == gpu_bytes[i]);
    }
  }

  // 2. kxq0/kxq1/numAtCurMin (deterministic from registers).
  double cpu_kxq0{}, cpu_kxq1{}, gpu_kxq0{}, gpu_kxq1{};
  std::memcpy(&cpu_kxq0, cpu_bytes.data() + KXQ0_OFF, sizeof(double));
  std::memcpy(&cpu_kxq1, cpu_bytes.data() + KXQ1_OFF, sizeof(double));
  std::memcpy(&gpu_kxq0, gpu_bytes.data() + KXQ0_OFF, sizeof(double));
  std::memcpy(&gpu_kxq1, gpu_bytes.data() + KXQ1_OFF, sizeof(double));
  std::uint32_t cpu_v{}, gpu_v{};
  std::memcpy(&cpu_v, cpu_bytes.data() + NUM_AT_MIN_OFF, sizeof(std::uint32_t));
  std::memcpy(&gpu_v, gpu_bytes.data() + NUM_AT_MIN_OFF, sizeof(std::uint32_t));
  REQUIRE(cpu_kxq0 == Catch::Approx(gpu_kxq0).epsilon(1e-12));
  REQUIRE(cpu_kxq1 == Catch::Approx(gpu_kxq1).epsilon(1e-12));
  REQUIRE(cpu_v == gpu_v);

  // 3. Estimates within 3-sigma RSE bound of n. Since registers are equal and
  // both sides apply Composite (CPU side via the patched-FLAGS path -- see
  // composite_finalizer_test.cpp), CPU and GPU estimates differ only by the
  // CPU's HIP estimator selection, so we compare against `n`.
  const double gpu_est = gpu.get_estimate(stream);
  const double bound   = three_sigma_bound(lgK);
  const double rel     = std::abs(gpu_est - static_cast<double>(n)) / static_cast<double>(n);
  INFO("lgK=" << int(lgK) << " n=" << n << " gpu_est=" << gpu_est << " rel=" << rel
              << " bound=" << bound);
  REQUIRE(rel < bound);
}

}  // namespace

TEST_CASE("CPU/GPU HLL_8 byte-level parity", "[parity]")
{
  // n large enough that CPU also lands in HLL mode densely; small enough to
  // exercise non-saturated registers.
  run(8, 10'000, 0xCAFEBABE08ULL);
  run(12, 100'000, 0xCAFEBABE12ULL);
  run(16, 500'000, 0xCAFEBABE16ULL);
}
