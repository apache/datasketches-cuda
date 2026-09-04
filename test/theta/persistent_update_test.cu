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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/detail/common/error.hpp>
#include <datasketches/cuda/detail/theta/persistent_update.cuh>

namespace {

using namespace datasketches::cuda::detail::theta;

constexpr std::size_t slots       = 8192;
constexpr auto load_limit         = max_occupied(slots);
constexpr auto dynamic_shared     = required_shared_bytes(slots);
constexpr auto retained_capacity  = persistent_update_config::retained_capacity;
constexpr auto block_output_slots = persistent_update_config::block_output_slots;

std::vector<std::uint64_t> expected_output(std::vector<std::uint64_t> input)
{
  input.erase(std::remove_if(input.begin(),
                             input.end(),
                             [](std::uint64_t value) { return value == 0 || value >= max_theta; }),
              input.end());
  std::sort(input.begin(), input.end());
  input.erase(std::unique(input.begin(), input.end()), input.end());

  std::vector<std::uint64_t> expected(block_output_slots, empty_key);
  const auto retained = std::min(input.size(), retained_capacity);
  std::copy_n(input.begin(), retained, expected.begin());
  if (input.size() > retained_capacity) { expected[retained_capacity] = input[retained_capacity]; }
  return expected;
}

void check_reducer(const std::vector<std::uint64_t>& input)
{
  thrust::device_vector<std::uint64_t> device_input = input;
  thrust::device_vector<std::uint64_t> output(block_output_slots, empty_key);
  device_state initial{max_theta, 0, 0};
  thrust::device_vector<device_state> state(1);
  DATASKETCHES_CUDA_TRY(cudaMemcpy(
    thrust::raw_pointer_cast(state.data()), &initial, sizeof(initial), cudaMemcpyHostToDevice));

  const auto kernel = persistent_update_kernel<decltype(device_input.begin()), identity_hash>;
  DATASKETCHES_CUDA_TRY(cudaFuncSetAttribute(
    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_shared)));
  kernel<<<1, persistent_update_config::max_block_threads, dynamic_shared>>>(
    device_input.begin(),
    device_input.size(),
    identity_hash{},
    thrust::raw_pointer_cast(state.data()),
    thrust::raw_pointer_cast(output.data()),
    slots,
    load_limit);
  DATASKETCHES_CUDA_TRY(cudaDeviceSynchronize());

  thrust::host_vector<std::uint64_t> result = output;
  std::sort(result.begin(), result.end());
  REQUIRE(std::vector<std::uint64_t>(result.begin(), result.end()) == expected_output(input));

  device_state final{};
  DATASKETCHES_CUDA_TRY(cudaMemcpy(
    &final, thrust::raw_pointer_cast(state.data()), sizeof(final), cudaMemcpyDeviceToHost));
  const auto expected = expected_output(input);
  REQUIRE(final.theta ==
          (expected[retained_capacity] == empty_key ? max_theta : expected[retained_capacity]));
}

std::vector<std::uint64_t> consecutive_input(std::size_t size)
{
  std::vector<std::uint64_t> input(size);
  std::iota(input.begin(), input.end(), std::uint64_t{1});
  return input;
}

}  // namespace

TEST_CASE("Theta persistent update retains the exact bottom-k", "[theta][persistent]")
{
  check_reducer(consecutive_input(20'000));

  auto duplicate_input = consecutive_input(20'000);
  for (std::size_t i = retained_capacity + 1; i < duplicate_input.size(); ++i) {
    duplicate_input[i] = duplicate_input[i % (retained_capacity + 1)];
  }
  duplicate_input[duplicate_input.size() - 2] = 0;
  duplicate_input[duplicate_input.size() - 1] = empty_key;
  check_reducer(duplicate_input);
}

TEST_CASE("Theta persistent update handles boundary-sized inputs", "[theta][persistent][boundary]")
{
  const std::vector<std::size_t> sizes{
    0,
    1,
    31,
    32,
    33,
    1023,
    1024,
    1025,
    retained_capacity - 1,
    retained_capacity,
    retained_capacity + 1,
    load_limit - 1,
    load_limit,
    load_limit + 17,
  };

  for (const auto size : sizes) {
    CAPTURE(size);
    check_reducer(consecutive_input(size));
  }
}

TEST_CASE("Theta persistent update is independent of arrival order",
          "[theta][persistent][ordering]")
{
  auto input = consecutive_input(20'003);
  std::mt19937_64 rng{0x5eedULL};
  std::shuffle(input.begin(), input.end(), rng);
  check_reducer(input);
}

TEST_CASE("Theta persistent update handles duplicates and reserved hashes",
          "[theta][persistent][dedup]")
{
  std::vector<std::uint64_t> input(15'007);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<std::uint64_t>(i % 5001) + 1;
  }
  input[13]   = 0;
  input[1024] = max_theta;
  input[4097] = empty_key;

  std::mt19937_64 rng{0xded0ULL};
  std::shuffle(input.begin(), input.end(), rng);
  check_reducer(input);
}
