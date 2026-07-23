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

#pragma once

#include <cstdint>
#include <cuda/std/cmath>

#include <cuda_runtime.h>

#include <datasketches/cuda/detail/hll/composite_interpolation_table.cuh>

namespace datasketches::cuda::detail::hll {

//! @brief Returns the HLL raw harmonic-mean estimate.
[[nodiscard]] __host__ __device__ inline double raw_estimate(double z, std::uint8_t lg_k) noexcept
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
  return (correction_factor * config_k * config_k) / z;
}

[[nodiscard]] __host__ __device__ inline double harmonic_number(std::uint32_t value) noexcept
{
  switch (value) {
    case 0: return 0.0;
    case 1: return 1.0;
    case 2: return 1.5;
    case 3: return 11.0 / 6.0;
    case 4: return 25.0 / 12.0;
    case 5: return 137.0 / 60.0;
    case 6: return 49.0 / 20.0;
    case 7: return 363.0 / 140.0;
    case 8: return 761.0 / 280.0;
    case 9: return 7129.0 / 2520.0;
    case 10: return 7381.0 / 2520.0;
    case 11: return 83711.0 / 27720.0;
    case 12: return 86021.0 / 27720.0;
    case 13: return 1145993.0 / 360360.0;
    case 14: return 1171733.0 / 360360.0;
    case 15: return 1195757.0 / 360360.0;
    case 16: return 2436559.0 / 720720.0;
    case 17: return 42142223.0 / 12252240.0;
    case 18: return 14274301.0 / 4084080.0;
    case 19: return 275295799.0 / 77597520.0;
    case 20: return 55835135.0 / 15519504.0;
    case 21: return 18858053.0 / 5173168.0;
    case 22: return 19093197.0 / 5173168.0;
    case 23: return 444316699.0 / 118982864.0;
    case 24: return 1347822955.0 / 356948592.0;
    default: break;
  }

  constexpr double euler_mascheroni = 0.577215664901532860606512090082;
  const double x                    = static_cast<double>(value);
  const double inv_sq               = 1.0 / (x * x);
  double sum                        = ::cuda::std::log(x) + euler_mascheroni + (1.0 / (2.0 * x));
  double power                      = inv_sq;
  sum -= power * (1.0 / 12.0);
  power *= inv_sq;
  sum += power * (1.0 / 120.0);
  power *= inv_sq;
  sum -= power * (1.0 / 252.0);
  power *= inv_sq;
  sum += power * (1.0 / 240.0);
  return sum;
}

//! @brief Returns the low-cardinality bitmap estimate.
[[nodiscard]] __host__ __device__ inline double bitmap_estimate(std::uint32_t num_zeroes,
                                                                std::uint8_t lg_k) noexcept
{
  const std::uint32_t config_k = 1u << lg_k;
  if (num_zeroes == 0) { return config_k * ::cuda::std::log(config_k / 0.5); }
  return config_k * (harmonic_number(config_k) - harmonic_number(num_zeroes));
}

[[nodiscard]] __host__ __device__ inline int find_straddle(const double* x_values,
                                                           int length,
                                                           double x) noexcept
{
  int left  = 0;
  int right = length - 1;
  while (left + 1 < right) {
    const int middle = left + ((right - left) / 2);
    if (x_values[middle] <= x) {
      left = middle;
    } else {
      right = middle;
    }
  }
  return left;
}

[[nodiscard]] __host__ __device__ inline double cubic_interpolate(double x0,
                                                                  double y0,
                                                                  double x1,
                                                                  double y1,
                                                                  double x2,
                                                                  double y2,
                                                                  double x3,
                                                                  double y3,
                                                                  double x) noexcept
{
  const double l0_numer = (x - x1) * (x - x2) * (x - x3);
  const double l1_numer = (x - x0) * (x - x2) * (x - x3);
  const double l2_numer = (x - x0) * (x - x1) * (x - x3);
  const double l3_numer = (x - x0) * (x - x1) * (x - x2);

  const double l0_denom = (x0 - x1) * (x0 - x2) * (x0 - x3);
  const double l1_denom = (x1 - x0) * (x1 - x2) * (x1 - x3);
  const double l2_denom = (x2 - x0) * (x2 - x1) * (x2 - x3);
  const double l3_denom = (x3 - x0) * (x3 - x1) * (x3 - x2);

  const double term0 = y0 * l0_numer / l0_denom;
  const double term1 = y1 * l1_numer / l1_denom;
  const double term2 = y2 * l2_numer / l2_denom;
  const double term3 = y3 * l3_numer / l3_denom;
  return term0 + term1 + term2 + term3;
}

[[nodiscard]] __host__ __device__ inline double interpolate_composite(const double* x_values,
                                                                      std::uint32_t y_stride,
                                                                      double x) noexcept
{
  constexpr int length = static_cast<int>(composite_interpolation::num_x_values);
  if (x == x_values[length - 1]) { return static_cast<double>(y_stride) * (length - 1); }

  const int offset = find_straddle(x_values, length, x);
  const int base   = offset == 0 ? 0 : (offset == length - 2 ? offset - 2 : offset - 1);

  return cubic_interpolate(x_values[base + 0],
                           static_cast<double>(y_stride) * (base + 0),
                           x_values[base + 1],
                           static_cast<double>(y_stride) * (base + 1),
                           x_values[base + 2],
                           static_cast<double>(y_stride) * (base + 2),
                           x_values[base + 3],
                           static_cast<double>(y_stride) * (base + 3),
                           x);
}

//! @brief DataSketches Composite cardinality estimate for an HLL_8 register
//! array represented by its harmonic sum and zero-register count.
[[nodiscard]] __host__ __device__ inline double composite_estimate(double z,
                                                                   std::uint32_t num_zeroes,
                                                                   std::uint8_t lg_k) noexcept
{
  const double raw_est         = raw_estimate(z, lg_k);
  const double* x_values       = composite_interpolation::x_values_for(lg_k);
  const std::uint32_t y_stride = composite_interpolation::y_stride_for(lg_k);

  if (raw_est < x_values[0]) { return 0.0; }

  constexpr std::uint32_t last = composite_interpolation::last_x_offset;
  if (raw_est > x_values[last]) {
    const double final_y = static_cast<double>(y_stride) * last;
    return raw_est * (final_y / x_values[last]);
  }

  const double adjusted = interpolate_composite(x_values, y_stride, raw_est);
  if (adjusted > static_cast<double>(3u << lg_k)) { return adjusted; }

  const double linear  = bitmap_estimate(num_zeroes, lg_k);
  const double average = (adjusted + linear) / 2.0;

  double crossover = 0.64;
  if (lg_k == 4) {
    crossover = 0.718;
  } else if (lg_k == 5) {
    crossover = 0.672;
  }

  return average > (crossover * static_cast<double>(1u << lg_k)) ? adjusted : linear;
}

}  // namespace datasketches::cuda::detail::hll
