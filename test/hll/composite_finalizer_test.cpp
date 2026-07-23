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

// Validates that `datasketches::cuda::detail::hll::composite_estimate` produces the
// same result as `datasketches::hll_sketch::get_estimate()` when the CPU sketch
// is forced into Composite mode (oooFlag=true).
//
// The test pulls (kxq0, kxq1, curMin, numAtCurMin) out of the CPU sketch's
// serialized wire format and feeds them to our finalizer. To make the CPU side
// also use Composite (instead of HIP), the FLAGS byte is patched to set the
// OOO bit before deserialize.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <hll.hpp>

#include <datasketches/cuda/detail/hll/composite_finalizer.cuh>
#include <datasketches/cuda/detail/hll/composite_interpolation_table.cuh>
#include <datasketches/cuda/detail/hll/relative_error.cuh>

#include <CompositeInterpolationXTable.hpp>
#include <CubicInterpolation.hpp>
#include <HarmonicNumbers.hpp>

namespace {

constexpr std::size_t HIP_ACCUM_OFF  = 8;
constexpr std::size_t KXQ0_OFF       = 16;
constexpr std::size_t KXQ1_OFF       = 24;
constexpr std::size_t NUM_AT_MIN_OFF = 32;
constexpr std::size_t FLAGS_OFF      = 5;
constexpr std::size_t CUR_MIN_OFF    = 6;
constexpr uint8_t OOO_FLAG_MASK      = 0x10;

struct cpu_state {
  double kxq0;
  double kxq1;
  uint8_t cur_min;
  uint32_t num_at_cur_min;
};

cpu_state extract_state(const std::vector<uint8_t>& bytes)
{
  cpu_state s{};
  std::memcpy(&s.kxq0, bytes.data() + KXQ0_OFF, sizeof(double));
  std::memcpy(&s.kxq1, bytes.data() + KXQ1_OFF, sizeof(double));
  s.cur_min = bytes[CUR_MIN_OFF];
  std::memcpy(&s.num_at_cur_min, bytes.data() + NUM_AT_MIN_OFF, sizeof(uint32_t));
  return s;
}

// Build an HLL_8 CPU sketch promoted into HLL mode by inserting `n` distinct
// keys via `start_full_size=true`, then return:
//   1. our composite_estimate result, and
//   2. CPU `get_estimate()` after patching the FLAGS byte to set OOO so that
//      the CPU also returns Composite (not HIP).
struct test_result {
  double our_estimate;
  double cpu_composite_estimate;
};

test_result run(uint8_t lgK, uint64_t n, uint64_t seed)
{
  ::datasketches::hll_sketch sketch(lgK, ::datasketches::HLL_8, /*start_full_size=*/true);
  std::mt19937_64 rng(seed);
  for (uint64_t i = 0; i < n; ++i) {
    const uint64_t k = rng();
    sketch.update(k);
  }

  auto bytes       = sketch.serialize_compact();
  const auto state = extract_state(bytes);

  // Patch FLAGS byte so the CPU re-deserialize hits the Composite path.
  bytes[FLAGS_OFF]   = static_cast<uint8_t>(bytes[FLAGS_OFF] | OOO_FLAG_MASK);
  auto cpu_composite = ::datasketches::hll_sketch::deserialize(bytes.data(), bytes.size());

  return {
    datasketches::cuda::detail::hll::composite_estimate(
      state.kxq0 + state.kxq1, state.num_at_cur_min, lgK),
    cpu_composite.get_estimate(),
  };
}

}  // namespace

TEST_CASE("composite_finalizer matches CPU getCompositeEstimate", "[composite_finalizer]")
{
  using Catch::Approx;
  for (std::uint8_t lgK = 4; lgK <= 18; ++lgK) {
    const std::uint64_t config_k = std::uint64_t{1} << lgK;
    const std::uint64_t high_n   = config_k * 32 < 200'000 ? config_k * 32 : 200'000;
    for (std::uint64_t n : {config_k / 4, config_k * 2, high_n}) {
      auto r = run(lgK, n, /*seed=*/0xC0FFEE0042 ^ (uint64_t(lgK) << 40) ^ n);
      INFO("lgK=" << int(lgK) << " n=" << n);
      REQUIRE(r.our_estimate == Approx(r.cpu_composite_estimate).epsilon(1e-12));
    }
  }
}

TEST_CASE("composite_finalizer empty sketch yields 0", "[composite_finalizer]")
{
  using Catch::Approx;
  // An empty sketch (all registers = 0) should land in the rawEst < xArr[0]
  // branch and return 0.
  const uint8_t lgK      = 12;
  const uint32_t configK = 1u << lgK;
  // For all-zero registers: kxq0+kxq1 = configK, raw = correctionFactor*configK, and
  // raw is well below xArr[0] for typical lgK.
  const double kxq_sum = static_cast<double>(configK);
  const double our =
    datasketches::cuda::detail::hll::composite_estimate(kxq_sum, /*num_zeroes=*/configK, lgK);
  REQUIRE(our == Approx(0.0));
}

TEST_CASE("host interpolation tables match DataSketches C++", "[composite_finalizer][table]")
{
  using Catch::Approx;
  namespace local = datasketches::cuda::detail::hll::composite_interpolation;

  for (std::uint8_t lg_k = local::min_lg_k; lg_k <= local::max_lg_k; ++lg_k) {
    const double* expected = ::datasketches::CompositeInterpolationXTable<>::get_x_arr(lg_k);
    const double* actual   = local::x_values_for(lg_k);
    REQUIRE(local::y_stride_for(lg_k) ==
            ::datasketches::CompositeInterpolationXTable<>::get_y_stride(lg_k));

    for (std::uint32_t i = 0; i < local::num_x_values; ++i) {
      CAPTURE(lg_k, i);
      REQUIRE(actual[i] == expected[i]);
    }

    const auto y_stride = local::y_stride_for(lg_k);
    for (std::uint32_t i = 0; i + 1 < local::num_x_values; ++i) {
      const double midpoint  = (actual[i] + actual[i + 1]) / 2.0;
      const double reference = ::datasketches::CubicInterpolation<>::usingXArrAndYStride(
        expected, static_cast<int>(local::num_x_values), y_stride, midpoint);
      const double result =
        datasketches::cuda::detail::hll::interpolate_composite(actual, y_stride, midpoint);
      CAPTURE(lg_k, i, midpoint);
      REQUIRE(result == Approx(reference).epsilon(1e-12));
    }
  }
}

TEST_CASE("host harmonic and bitmap estimators match DataSketches C++",
          "[composite_finalizer][bitmap]")
{
  using Catch::Approx;
  for (std::uint8_t lg_k = 4; lg_k <= 18; ++lg_k) {
    const std::uint32_t config_k = 1u << lg_k;
    for (std::uint32_t zeroes : {config_k, config_k / 2, 1u}) {
      const auto hits        = config_k - zeroes;
      const double reference = ::datasketches::HarmonicNumbers<>::getBitMapEstimate(config_k, hits);
      CAPTURE(lg_k, zeroes);
      REQUIRE(datasketches::cuda::detail::hll::bitmap_estimate(zeroes, lg_k) ==
              Approx(reference).epsilon(1e-15));
    }
    REQUIRE(datasketches::cuda::detail::hll::bitmap_estimate(0, lg_k) ==
            Approx(config_k * std::log(config_k / 0.5)).epsilon(1e-15));
  }
}

TEST_CASE("non-HIP relative errors match DataSketches C++", "[relative_error]")
{
  using Catch::Approx;
  for (std::uint8_t lg_k = 4; lg_k <= 18; ++lg_k) {
    for (std::uint8_t num_std_dev = 1; num_std_dev <= 3; ++num_std_dev) {
      for (bool upper_bound : {false, true}) {
        const double expected =
          ::datasketches::HllUtil<>::getRelErr(upper_bound, /*unioned=*/true, lg_k, num_std_dev);
        const double actual =
          datasketches::cuda::detail::hll::relative_error(upper_bound, lg_k, num_std_dev);
        CAPTURE(lg_k, num_std_dev, upper_bound);
        REQUIRE(actual == Approx(expected).epsilon(1e-15));
      }
    }
  }
}
