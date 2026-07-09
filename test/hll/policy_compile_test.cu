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

// Compile-only sanity check that `datasketches::cuda::detail::hll::policy`
// satisfies the cudax `_Policy` concept and can be substituted for the default
// policy in `cuda::experimental::cuco::hyperloglog` and `hyperloglog_ref`.
//
// No GPU is required; the test asserts that the type instantiations compile and
// that the expected member typedefs and method signatures exist.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda/experimental/__cuco/hyperloglog.cuh>
#include <cuda/experimental/__cuco/hyperloglog_ref.cuh>

#include <catch2/catch_test_macros.hpp>

#include <common_defs.hpp>

#include <datasketches/cuda/detail/hll/policy.cuh>
#include <datasketches/cuda/detail/hll/reduction_state.hpp>

using Key      = ::std::int64_t;
using policy_t = datasketches::cuda::detail::hll::policy<Key>;
using hll_t    = cuda::experimental::cuco::
  hyperloglog<Key, ::cuda::device_memory_pool_ref, ::cuda::thread_scope_device, policy_t>;
using hll_ref_t =
  cuda::experimental::cuco::hyperloglog_ref<Key, ::cuda::thread_scope_device, policy_t>;

static_assert(std::is_same_v<policy_t::hash_result_type, __uint128_t>,
              "policy::hash_result_type must be __uint128_t");
static_assert(std::is_same_v<policy_t::register_type, ::std::int32_t>,
              "policy::register_type must be int32_t");
static_assert(
  std::is_same_v<datasketches::cuda::detail::hll::register_type, policy_t::register_type>,
  "detail::hll::register_type must match policy::register_type");
static_assert(policy_t::default_seed == datasketches::DEFAULT_SEED,
              "policy::default_seed must stay pinned to datasketches-cpp DEFAULT_SEED "
              "(common_defs.hpp:34)");
static_assert(datasketches::cuda::detail::hll::default_seed == datasketches::DEFAULT_SEED,
              "project-wide default_seed must stay pinned to datasketches-cpp DEFAULT_SEED");

TEST_CASE("policy satisfies cudax concept", "[policy][compile]")
{
  // The static_asserts above guarantee compile-time conformance. This runtime
  // check is just here to give Catch2 something to discover and execute.
  REQUIRE(policy_t::default_seed == datasketches::DEFAULT_SEED);
  static_cast<void>(sizeof(hll_t));
  static_cast<void>(sizeof(hll_ref_t));
}

TEST_CASE("register_index is low precision bits of h1", "[policy][bit_slicing]")
{
  policy_t policy{};
  // h1 = 0xCAFEBABE_DEADBEEF, h2 = 0xFEEDFACE_BAADF00D
  const __uint128_t h = (static_cast<__uint128_t>(0xFEEDFACEBAADF00DULL) << 64) |
                        static_cast<__uint128_t>(0xCAFEBABEDEADBEEFULL);
  // For precision=12, mask = 0xFFF; expect low 12 bits of h1 = 0xEEF
  REQUIRE(policy.register_index(h, 12) == 0xEEFu);
  // For precision=21 (max lgK), mask = 0x1FFFFF; expect low 21 bits of h1
  REQUIRE(policy.register_index(h, 21) == (0xCAFEBABEDEADBEEFULL & 0x1FFFFFu));
}

TEST_CASE("register_value is min(clz(h2), 62) + 1", "[policy][bit_slicing]")
{
  policy_t policy{};

  // h2 = 1 << 63 (msb set) -> clz=0, rho = 1
  __uint128_t h_msb = static_cast<__uint128_t>(1ULL << 63) << 64;
  REQUIRE(policy.register_value(h_msb, 0) == 1u);

  // h2 = 0 -> clz=64, capped to 62, rho = 63
  __uint128_t h_zero = static_cast<__uint128_t>(0);
  REQUIRE(policy.register_value(h_zero, 0) == 63u);

  // h2 = 1 -> clz=63, capped to 62, rho = 63
  __uint128_t h_one = static_cast<__uint128_t>(1ULL) << 64;
  REQUIRE(policy.register_value(h_one, 0) == 63u);

  // h2 = 1 << 60 -> clz=3, rho = 4
  __uint128_t h_sixty = static_cast<__uint128_t>(1ULL << 60) << 64;
  REQUIRE(policy.register_value(h_sixty, 0) == 4u);
}
