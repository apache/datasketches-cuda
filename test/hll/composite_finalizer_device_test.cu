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
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/detail/hll/composite_finalizer.cuh>
#include <datasketches/cuda/detail/hll/composite_interpolation_table.cuh>

namespace {

struct finalizer_case {
  double z;
  std::uint32_t zeroes;
  std::uint8_t lg_k;
};

__global__ void finalize_kernel(const finalizer_case* cases, double* results, std::size_t size)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    const auto c   = cases[index];
    results[index] = datasketches::cuda::detail::hll::composite_estimate(c.z, c.zeroes, c.lg_k);
  }
}

double z_for_raw(double raw, std::uint8_t lg_k)
{
  const std::uint32_t config_k = 1u << lg_k;
  double correction_factor;
  if (lg_k == 4) {
    correction_factor = 0.673;
  } else if (lg_k == 5) {
    correction_factor = 0.697;
  } else if (lg_k == 6) {
    correction_factor = 0.709;
  } else {
    correction_factor = 0.7213 / (1.0 + (1.079 / config_k));
  }
  return (correction_factor * config_k * config_k) / raw;
}

}  // namespace

TEST_CASE("Composite finalizer host and device results agree", "[composite_finalizer][device]")
{
  namespace table = datasketches::cuda::detail::hll::composite_interpolation;
  std::vector<finalizer_case> cases;

  for (std::uint8_t lg_k = table::min_lg_k; lg_k <= table::max_lg_k; ++lg_k) {
    const auto* x_values = table::x_values_for(lg_k);
    const auto config_k  = std::uint32_t{1} << lg_k;
    cases.push_back({static_cast<double>(config_k), config_k, lg_k});
    cases.push_back({z_for_raw(x_values[0] * 0.5, lg_k), config_k, lg_k});
    cases.push_back({z_for_raw(x_values[table::last_x_offset] * 2.0, lg_k), 0, lg_k});

    for (std::uint32_t i = 0; i + 1 < table::num_x_values; ++i) {
      const double midpoint = (x_values[i] + x_values[i + 1]) / 2.0;
      cases.push_back({z_for_raw(midpoint, lg_k), config_k / 2, lg_k});
    }
  }

  std::vector<double> expected(cases.size());
  std::transform(cases.begin(), cases.end(), expected.begin(), [](const finalizer_case& c) {
    return datasketches::cuda::detail::hll::composite_estimate(c.z, c.zeroes, c.lg_k);
  });

  thrust::device_vector<finalizer_case> device_cases = cases;
  thrust::device_vector<double> device_results(cases.size());
  constexpr int block_size = 256;
  const int grid_size      = static_cast<int>((cases.size() + block_size - 1) / block_size);
  finalize_kernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(device_cases.data()),
                                             thrust::raw_pointer_cast(device_results.data()),
                                             cases.size());
  REQUIRE(cudaGetLastError() == cudaSuccess);

  std::vector<double> actual(device_results.size());
  thrust::copy(device_results.begin(), device_results.end(), actual.begin());
  for (std::size_t i = 0; i < cases.size(); ++i) {
    CAPTURE(i, cases[i].lg_k, cases[i].zeroes, cases[i].z, expected[i], actual[i]);
    REQUIRE(actual[i] == Catch::Approx(expected[i]).epsilon(1e-12).margin(1e-9));
  }
}
