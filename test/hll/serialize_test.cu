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

// GPU-side serialize: produce bytes, hand them to CPU `datasketches::hll_sketch`,
// confirm CPU's get_estimate matches GPU's get_estimate.

#include <cmath>
#include <cstdint>
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

TEST_CASE("GPU serialize bytes accepted by CPU deserialize", "[serialize]")
{
  using Catch::Approx;
  for (uint8_t lgK : {uint8_t{8}, uint8_t{12}, uint8_t{16}}) {
    const uint64_t n = (uint64_t{1} << lgK) * 64;  // saturate well past LIST/SET threshold

    std::vector<uint64_t> host_keys(n);
    std::mt19937_64 rng(0xFACEB00CULL ^ lgK);
    for (auto& k : host_keys)
      k = rng();
    thrust::device_vector<uint64_t> dev_keys = host_keys;

    ::cuda::stream stream{::cuda::devices[0]};
    auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
    datasketches::cuda::hll_sketch<uint64_t> gpu(stream, mr, lgK);
    gpu.update(stream, dev_keys.begin(), dev_keys.end());

    auto bytes = gpu.serialize_compact(stream);
    REQUIRE(bytes.size() == 40u + (1u << lgK));

    auto cpu = ::datasketches::hll_sketch::deserialize(bytes.data(), bytes.size());
    INFO("lgK=" << int(lgK) << " n=" << n);
    REQUIRE(cpu.get_estimate() == Approx(gpu.get_estimate(stream)).epsilon(1e-12));
  }
}

TEST_CASE("GPU serialize -> GPU deserialize round-trip", "[serialize]")
{
  using Catch::Approx;
  const uint8_t lgK = 12;
  const uint64_t n  = 200'000;
  std::vector<uint64_t> host_keys(n);
  std::mt19937_64 rng(0xDEFEC8EDULL);
  for (auto& k : host_keys)
    k = rng();
  thrust::device_vector<uint64_t> dev_keys = host_keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::hll_sketch<uint64_t> a(stream, mr, lgK);
  a.update(stream, dev_keys.begin(), dev_keys.end());

  auto bytes = a.serialize_compact(stream);
  auto b     = datasketches::cuda::hll_sketch<uint64_t>::deserialize(
    stream, ::cuda::std::span<const std::uint8_t>{bytes.data(), bytes.size()}, mr);

  REQUIRE(a.get_estimate(stream) == Approx(b.get_estimate(stream)));
  REQUIRE(a.serialize_compact(stream) == b.serialize_compact(stream));  // byte-equal
}
