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
