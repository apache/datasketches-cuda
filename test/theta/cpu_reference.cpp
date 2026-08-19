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
std::vector<std::uint8_t> cpu_image(const std::vector<T>& values,
                                    std::uint8_t lg_k,
                                    std::uint64_t seed,
                                    float p)
{
  auto sketch =
    ::datasketches::update_theta_sketch::builder().set_lg_k(lg_k).set_seed(seed).set_p(p).build();
  for (const auto value : values)
    sketch.update(value);
  sketch.trim();
  return sketch.compact(true).serialize();
}

compact_metadata cpu_metadata(const std::vector<std::uint8_t>& bytes, std::uint64_t seed)
{
  const auto sketch =
    ::datasketches::compact_theta_sketch::deserialize(bytes.data(), bytes.size(), seed);
  return compact_metadata{
    sketch.is_empty(), sketch.is_ordered(), sketch.get_num_retained(), sketch.get_theta64()};
}

set_operation_images cpu_set_operation_images(const std::vector<std::uint64_t>& a,
                                              const std::vector<std::uint64_t>& b,
                                              std::uint8_t lg_k,
                                              std::uint64_t seed)
{
  auto a_update =
    ::datasketches::update_theta_sketch::builder().set_lg_k(lg_k).set_seed(seed).build();
  auto b_update =
    ::datasketches::update_theta_sketch::builder().set_lg_k(lg_k).set_seed(seed).build();
  for (const auto value : a)
    a_update.update(value);
  for (const auto value : b)
    b_update.update(value);
  a_update.trim();
  b_update.trim();
  const auto a_compact = a_update.compact(true);
  const auto b_compact = b_update.compact(true);

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

template std::vector<std::uint8_t> cpu_image(const std::vector<std::uint64_t>&,
                                             std::uint8_t,
                                             std::uint64_t,
                                             float);
template std::vector<std::uint8_t> cpu_image(const std::vector<std::int64_t>&,
                                             std::uint8_t,
                                             std::uint64_t,
                                             float);
template std::vector<std::uint8_t> cpu_image(const std::vector<std::uint32_t>&,
                                             std::uint8_t,
                                             std::uint64_t,
                                             float);
template std::vector<std::uint8_t> cpu_image(const std::vector<std::int32_t>&,
                                             std::uint8_t,
                                             std::uint64_t,
                                             float);
template std::vector<std::uint8_t> cpu_image(const std::vector<std::uint16_t>&,
                                             std::uint8_t,
                                             std::uint64_t,
                                             float);
template std::vector<std::uint8_t> cpu_image(const std::vector<std::int16_t>&,
                                             std::uint8_t,
                                             std::uint64_t,
                                             float);
template std::vector<std::uint8_t> cpu_image(const std::vector<std::uint8_t>&,
                                             std::uint8_t,
                                             std::uint64_t,
                                             float);
template std::vector<std::uint8_t> cpu_image(const std::vector<std::int8_t>&,
                                             std::uint8_t,
                                             std::uint64_t,
                                             float);
template std::vector<std::uint8_t> cpu_image(const std::vector<double>&,
                                             std::uint8_t,
                                             std::uint64_t,
                                             float);
template std::vector<std::uint8_t> cpu_image(const std::vector<float>&,
                                             std::uint8_t,
                                             std::uint64_t,
                                             float);

}  // namespace theta_test
