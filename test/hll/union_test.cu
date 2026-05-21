// Verifies merge: two sketches with disjoint key streams, when merged, produce
// an estimate close to the union cardinality.

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/hll.hpp>

TEST_CASE("merge of two disjoint sketches estimates union cardinality", "[hll_sketch][union]")
{
  const uint8_t lgK      = 12;
  const uint64_t n_each  = 50'000;
  const uint64_t n_total = 2 * n_each;

  std::vector<uint64_t> a_keys(n_each), b_keys(n_each);
  std::mt19937_64 rng(0xC0FFEE0042ULL);
  for (uint64_t i = 0; i < n_each; ++i) {
    a_keys[i] = rng();
    b_keys[i] = rng();
  }

  thrust::device_vector<uint64_t> a_dev = a_keys;
  thrust::device_vector<uint64_t> b_dev = b_keys;

  datasketches::cuda::hll_sketch<uint64_t> a(lgK);
  datasketches::cuda::hll_sketch<uint64_t> b(lgK);
  a.update(a_dev.begin(), a_dev.end());
  b.update(b_dev.begin(), b_dev.end());

  a.merge(b);

  const double est   = a.get_estimate();
  const double bound = 3.0 * 1.04 / std::sqrt(static_cast<double>(1u << lgK));
  const double rel   = std::abs(est - static_cast<double>(n_total)) / static_cast<double>(n_total);
  INFO("lgK=" << int(lgK) << " expected=" << n_total << " estimate=" << est << " rel_err=" << rel
              << " bound=" << bound);
  REQUIRE(rel < bound);
}
