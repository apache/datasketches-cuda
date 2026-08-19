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
#include <cuda/stream>
#include <limits>
#include <random>
#include <vector>

#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/theta.hpp>

#include "cpu_reference.hpp"

namespace {

template <class T>
void compare_typed(const std::vector<T>& values)
{
  constexpr std::uint8_t lg_k     = 12;
  thrust::device_vector<T> device = values;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<T> gpu(stream, mr, lg_k);
  gpu.update(stream, device.begin(), device.end());
  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(values, lg_k, 9001));
}

template <class T>
std::vector<T> random_integrals(std::size_t count, std::uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::vector<T> values(count);
  for (auto& value : values)
    value = static_cast<T>(rng());
  return values;
}

}  // namespace

TEST_CASE("Theta primitive integer hashing matches CPU", "[theta][parity][typed]")
{
  compare_typed(random_integrals<std::uint64_t>(3000, 1));
  compare_typed(random_integrals<std::int64_t>(3000, 2));
  compare_typed(random_integrals<std::uint32_t>(3000, 3));
  compare_typed(random_integrals<std::int32_t>(3000, 4));
  compare_typed(random_integrals<std::uint16_t>(3000, 5));
  compare_typed(random_integrals<std::int16_t>(3000, 6));
  compare_typed(random_integrals<std::uint8_t>(3000, 7));
  compare_typed(random_integrals<std::int8_t>(3000, 8));
}

TEST_CASE("Theta floating-point normalization matches CPU", "[theta][parity][typed]")
{
  std::mt19937_64 rng(9);
  std::uniform_real_distribution<double> doubles(-1e12, 1e12);
  std::uniform_real_distribution<float> floats(-1e6F, 1e6F);

  std::vector<double> double_values;
  std::vector<float> float_values;
  for (int i = 0; i < 3000; ++i) {
    double_values.push_back(doubles(rng));
    float_values.push_back(floats(rng));
  }
  double_values.push_back(0.0);
  double_values.push_back(-0.0);
  double_values.push_back(std::numeric_limits<double>::quiet_NaN());
  float_values.push_back(0.0F);
  float_values.push_back(-0.0F);
  float_values.push_back(std::numeric_limits<float>::quiet_NaN());

  compare_typed(double_values);
  compare_typed(float_values);
}
