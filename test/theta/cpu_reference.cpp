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

#include "cpu_reference.hpp"

#include <theta_a_not_b.hpp>
#include <theta_intersection.hpp>
#include <theta_sketch.hpp>
#include <theta_union.hpp>

namespace theta_test {

template <class T>
auto cpu_compact(const std::vector<T>& values,
                 std::uint8_t lg_k,
                 std::uint64_t seed,
                 float p = 1.0F)
{
  auto sketch =
    ::datasketches::update_theta_sketch::builder().set_lg_k(lg_k).set_seed(seed).set_p(p).build();
  for (const auto value : values)
    sketch.update(value);
  sketch.trim();
  return sketch.compact(true);
}

template <class T>
std::vector<std::uint8_t> cpu_image(const std::vector<T>& values,
                                    std::uint8_t lg_k,
                                    std::uint64_t seed,
                                    float p)
{
  return cpu_compact(values, lg_k, seed, p).serialize();
}

set_operation_images cpu_set_operation_images(const std::vector<std::uint64_t>& a,
                                              const std::vector<std::uint64_t>& b,
                                              std::uint8_t lg_k,
                                              std::uint64_t seed)
{
  const auto a_compact = cpu_compact(a, lg_k, seed);
  const auto b_compact = cpu_compact(b, lg_k, seed);

  auto set_union = ::datasketches::theta_union::builder().set_lg_k(lg_k).set_seed(seed).build();
  set_union.update(a_compact);
  set_union.update(b_compact);

  ::datasketches::theta_intersection intersection(seed);
  intersection.update(a_compact);
  intersection.update(b_compact);

  ::datasketches::theta_a_not_b a_not_b(seed);
  return set_operation_images{set_union.get_result(true).serialize(),
                              intersection.get_result(true).serialize(),
                              a_not_b.compute(a_compact, b_compact, true).serialize()};
}

// Keep upstream Theta templates out of CUDA translation units: NVCC mishandles
// their scalar pseudo-destructor calls.
#define INSTANTIATE_CPU_IMAGE(T)                \
  template std::vector<std::uint8_t> cpu_image( \
    const std::vector<T>&, std::uint8_t, std::uint64_t, float);

INSTANTIATE_CPU_IMAGE(std::uint64_t)
INSTANTIATE_CPU_IMAGE(std::int64_t)
INSTANTIATE_CPU_IMAGE(std::uint32_t)
INSTANTIATE_CPU_IMAGE(std::int32_t)
INSTANTIATE_CPU_IMAGE(std::uint16_t)
INSTANTIATE_CPU_IMAGE(std::int16_t)
INSTANTIATE_CPU_IMAGE(std::uint8_t)
INSTANTIATE_CPU_IMAGE(std::int8_t)
INSTANTIATE_CPU_IMAGE(double)
INSTANTIATE_CPU_IMAGE(float)
#undef INSTANTIATE_CPU_IMAGE

}  // namespace theta_test
