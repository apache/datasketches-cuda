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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace theta_test {

template <class T>
std::vector<std::uint8_t> cpu_image(const std::vector<T>& values,
                                    std::uint8_t lg_k,
                                    std::uint64_t seed,
                                    float p = 1.0F);

struct compact_metadata {
  bool empty;
  bool ordered;
  std::size_t retained;
  std::uint64_t theta;
};

compact_metadata cpu_metadata(const std::vector<std::uint8_t>& bytes, std::uint64_t seed = 9001);

struct set_operation_images {
  std::vector<std::uint8_t> set_union;
  std::vector<std::uint8_t> intersection;
  std::vector<std::uint8_t> a_not_b;
};

set_operation_images cpu_set_operation_images(const std::vector<std::uint64_t>& a,
                                              const std::vector<std::uint64_t>& b,
                                              std::uint8_t lg_k,
                                              std::uint64_t seed = 9001);

extern template std::vector<std::uint8_t> cpu_image(const std::vector<std::uint64_t>&,
                                                    std::uint8_t,
                                                    std::uint64_t,
                                                    float);
extern template std::vector<std::uint8_t> cpu_image(const std::vector<std::int64_t>&,
                                                    std::uint8_t,
                                                    std::uint64_t,
                                                    float);
extern template std::vector<std::uint8_t> cpu_image(const std::vector<std::uint32_t>&,
                                                    std::uint8_t,
                                                    std::uint64_t,
                                                    float);
extern template std::vector<std::uint8_t> cpu_image(const std::vector<std::int32_t>&,
                                                    std::uint8_t,
                                                    std::uint64_t,
                                                    float);
extern template std::vector<std::uint8_t> cpu_image(const std::vector<std::uint16_t>&,
                                                    std::uint8_t,
                                                    std::uint64_t,
                                                    float);
extern template std::vector<std::uint8_t> cpu_image(const std::vector<std::int16_t>&,
                                                    std::uint8_t,
                                                    std::uint64_t,
                                                    float);
extern template std::vector<std::uint8_t> cpu_image(const std::vector<std::uint8_t>&,
                                                    std::uint8_t,
                                                    std::uint64_t,
                                                    float);
extern template std::vector<std::uint8_t> cpu_image(const std::vector<std::int8_t>&,
                                                    std::uint8_t,
                                                    std::uint64_t,
                                                    float);
extern template std::vector<std::uint8_t> cpu_image(const std::vector<double>&,
                                                    std::uint8_t,
                                                    std::uint64_t,
                                                    float);
extern template std::vector<std::uint8_t> cpu_image(const std::vector<float>&,
                                                    std::uint8_t,
                                                    std::uint64_t,
                                                    float);

}  // namespace theta_test
