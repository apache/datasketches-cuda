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

// Validates that `datasketches::cuda::detail::reduce_hll8` produces the same
// (kxq0, kxq1, curMin, numAtCurMin) as the CPU sketch's stored state for the
// same register array. Pure host test.

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <hll.hpp>

#include <datasketches/cuda/detail/hll/reduction_state.hpp>

namespace {

constexpr std::size_t KXQ0_OFF       = 16;
constexpr std::size_t KXQ1_OFF       = 24;
constexpr std::size_t NUM_AT_MIN_OFF = 32;
constexpr std::size_t CUR_MIN_OFF    = 6;
constexpr std::size_t REG_OFF        = 40;

struct cpu_state {
  double kxq0;
  double kxq1;
  std::uint8_t cur_min;
  std::uint32_t num_at_cur_min;
};

cpu_state extract(const std::vector<uint8_t>& bytes)
{
  cpu_state s{};
  std::memcpy(&s.kxq0, bytes.data() + KXQ0_OFF, sizeof(double));
  std::memcpy(&s.kxq1, bytes.data() + KXQ1_OFF, sizeof(double));
  s.cur_min = bytes[CUR_MIN_OFF];
  std::memcpy(&s.num_at_cur_min, bytes.data() + NUM_AT_MIN_OFF, sizeof(std::uint32_t));
  return s;
}

// Returns the CPU's wire-format state and `reduce_hll8` applied to the same
// register array (widened to int32_t per cudax storage layout).
struct test_pair {
  cpu_state cpu;
  ::datasketches::cuda::detail::reduction_result gpu_side;
};

test_pair run(uint8_t lgK, uint64_t n, uint64_t seed)
{
  ::datasketches::hll_sketch sketch(lgK, ::datasketches::HLL_8, /*start_full_size=*/true);
  std::mt19937_64 rng(seed);
  for (uint64_t i = 0; i < n; ++i) {
    sketch.update(rng());
  }
  auto bytes = sketch.serialize_compact();

  const std::uint32_t configK = 1u << lgK;
  std::vector<std::int32_t> registers(configK);
  for (std::uint32_t i = 0; i < configK; ++i) {
    registers[i] = static_cast<std::int32_t>(bytes[REG_OFF + i]);
  }

  return {
    extract(bytes),
    ::datasketches::cuda::detail::reduce_hll8(
      ::cuda::std::span<const std::int32_t>{registers.data(), registers.size()}, lgK),
  };
}

}  // namespace

TEST_CASE("reduce_hll8 matches CPU stored kxq/curMin/numAtCurMin", "[reduction_state]")
{
  using Catch::Approx;
  for (uint8_t lgK : {uint8_t{8}, uint8_t{12}, uint8_t{16}}) {
    for (uint64_t n : {uint64_t{50}, uint64_t{1'000}, uint64_t{100'000}, uint64_t{2'000'000}}) {
      auto p = run(lgK, n, /*seed=*/0xBADCAFE042 ^ (uint64_t(lgK) << 40) ^ n);
      INFO("lgK=" << int(lgK) << " n=" << n);
      REQUIRE(p.gpu_side.kxq0 == Approx(p.cpu.kxq0).epsilon(1e-12));
      REQUIRE(p.gpu_side.kxq1 == Approx(p.cpu.kxq1).epsilon(1e-12));
      REQUIRE(p.gpu_side.cur_min == p.cpu.cur_min);
      REQUIRE(p.gpu_side.num_at_cur_min == p.cpu.num_at_cur_min);
    }
  }
}

TEST_CASE("reduce_hll8 empty register array", "[reduction_state]")
{
  const uint8_t lgK           = 12;
  const std::uint32_t configK = 1u << lgK;
  std::vector<std::int32_t> zeros(configK, 0);
  auto r = ::datasketches::cuda::detail::reduce_hll8(
    ::cuda::std::span<const std::int32_t>{zeros.data(), zeros.size()}, lgK);
  REQUIRE(r.kxq0 == static_cast<double>(configK));
  REQUIRE(r.kxq1 == 0.0);
  REQUIRE(r.cur_min == 0);
  REQUIRE(r.num_at_cur_min == configK);
}

TEST_CASE("reduce_hll8 all-saturated register array (rho=63)", "[reduction_state]")
{
  using Catch::Approx;
  const uint8_t lgK           = 8;
  const std::uint32_t configK = 1u << lgK;
  std::vector<std::int32_t> sat(configK, 63);
  auto r = ::datasketches::cuda::detail::reduce_hll8(
    ::cuda::std::span<const std::int32_t>{sat.data(), sat.size()}, lgK);
  REQUIRE(r.kxq0 == Approx(static_cast<double>(configK)));
  // Every register contributes (2^-63 - 1) to kxq1 (since 63 >= 32).
  // kxq1 = configK * (2^-63 - 1) -- a large negative number plus configK*2^-63.
  const double expected_kxq1 = configK * (::datasketches::INVERSE_POWERS_OF_2[63] - 1.0);
  REQUIRE(r.kxq1 == Approx(expected_kxq1));
  // For HLL_8 we hold `cur_min == 0` even when all registers are non-zero;
  // matches the CPU wire-format invariant.
  REQUIRE(r.cur_min == 0);
  REQUIRE(r.num_at_cur_min == 0);
}
