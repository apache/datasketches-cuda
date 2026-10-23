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
#include <vector>

#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/theta.hpp>

#include "cpu_reference.hpp"

namespace {

using sketch_type = datasketches::cuda::theta_sketch<std::uint64_t>;

std::vector<std::uint64_t> sequence(std::uint64_t first, std::uint64_t last)
{
  std::vector<std::uint64_t> values(last - first);
  for (std::uint64_t i = first; i < last; ++i)
    values[i - first] = i;
  return values;
}

sketch_type clone(::cuda::stream_ref stream,
                  ::cuda::device_memory_pool_ref mr,
                  const sketch_type& source,
                  std::uint8_t lg_k)
{
  const auto bytes = source.serialize_compact(stream);
  return sketch_type::deserialize(
    stream, ::cuda::std::span<const std::uint8_t>{bytes.data(), bytes.size()}, mr, lg_k);
}

}  // namespace

TEST_CASE("Theta set operations reject seed mismatch", "[theta][setops][seed]")
{
  auto values                                 = sequence(0, 100);
  thrust::device_vector<std::uint64_t> device = values;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  sketch_type a(stream, mr, 12, 1);
  sketch_type b(stream, mr, 12, 2);
  a.update(stream, device.begin(), device.end());
  b.update(stream, device.begin(), device.end());

  REQUIRE_THROWS_AS(a.merge(stream, b), std::invalid_argument);
  REQUIRE_THROWS_AS(a.intersect(stream, b), std::invalid_argument);
  REQUIRE_THROWS_AS(a.a_not_b(stream, b), std::invalid_argument);
}

TEST_CASE("Theta empty sampled union applies its configured p", "[theta][setops][sampling][parity]")
{
  constexpr std::uint8_t lg_k                 = 12;
  constexpr std::uint64_t seed                = 0xdecafbadULL;
  constexpr float p                           = 0.125F;
  auto values                                 = sequence(0, 10000);
  thrust::device_vector<std::uint64_t> device = values;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  sketch_type sampled_union(stream, mr, lg_k, seed, p);
  sketch_type source(stream, mr, lg_k, seed);
  source.update(stream, device.begin(), device.end());

  sampled_union.merge(stream, source);
  REQUIRE(sampled_union.serialize_compact(stream) == theta_test::cpu_image(values, lg_k, seed, p));
}

TEST_CASE("Theta set operations match CPU compact bytes", "[theta][setops][parity][serialization]")
{
  std::uint8_t lg_k = 12;
  std::vector<std::uint64_t> a_values;
  std::vector<std::uint64_t> b_values;
  SECTION("exact overlap")
  {
    a_values = sequence(0, 1000);
    b_values = sequence(500, 1500);
  }
  SECTION("exact disjoint")
  {
    a_values = sequence(0, 100);
    b_values = sequence(100, 200);
  }
  SECTION("estimated overlap")
  {
    lg_k     = 8;
    a_values = sequence(0, 10000);
    b_values = sequence(5000, 15000);
  }
  const auto cpu = theta_test::cpu_set_operation_images(a_values, b_values, lg_k);

  thrust::device_vector<std::uint64_t> a_device = a_values;
  thrust::device_vector<std::uint64_t> b_device = b_values;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  sketch_type a(stream, mr, lg_k);
  sketch_type b(stream, mr, lg_k);
  a.update(stream, a_device.begin(), a_device.end());
  b.update(stream, b_device.begin(), b_device.end());

  auto set_union = clone(stream, mr, a, lg_k);
  set_union.merge(stream, b);
  REQUIRE(set_union.serialize_compact(stream) == cpu.set_union);

  auto intersection = clone(stream, mr, a, lg_k);
  intersection.intersect(stream, b);
  REQUIRE(intersection.serialize_compact(stream) == cpu.intersection);

  auto difference = clone(stream, mr, a, lg_k);
  difference.a_not_b(stream, b);
  REQUIRE(difference.serialize_compact(stream) == cpu.a_not_b);
}
