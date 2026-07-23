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
#include <type_traits>

#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/hll_ref.cuh>

using device_ref = datasketches::cuda::hll_sketch_ref<std::uint64_t, ::cuda::thread_scope_device>;
using block_ref  = datasketches::cuda::hll_sketch_ref<std::uint64_t, ::cuda::thread_scope_block>;
using device_impl =
  datasketches::cuda::detail::hll::sketch_ref_impl<std::uint64_t, ::cuda::thread_scope_device>;

static_assert(std::is_same_v<device_ref::rebind_scope<::cuda::thread_scope_block>, block_ref>);
static_assert(std::is_same_v<device_ref::key_type, std::uint64_t>);
static_assert(std::is_same_v<device_ref::register_type, std::int32_t>);
static_assert(std::is_trivially_copyable_v<device_ref>);
static_assert(std::is_trivially_copyable_v<device_impl>);
static_assert(sizeof(device_ref) == sizeof(device_impl));

TEST_CASE("hll_ref public header is self-contained", "[hll_ref][compile]")
{
  REQUIRE(device_ref::sketch_alignment() == alignof(device_ref::register_type));
}
