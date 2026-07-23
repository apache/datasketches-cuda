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
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/stream>

#include <cuda_runtime.h>

#include <cuda/experimental/__cuco/detail/hyperloglog/hyperloglog_impl.cuh>
#include <cuda/experimental/__cuco/hyperloglog_ref.cuh>

#include <hll.hpp>

#include <datasketches/cuda/detail/hll/policy.cuh>
#include <datasketches/cuda/detail/hll/relative_error.cuh>

#include <cooperative_groups.h>

namespace datasketches::cuda::detail::hll {

// Shared non-owning implementation used by both public HLL handles.
template <class Key, ::cuda::thread_scope Scope = ::cuda::thread_scope_device>
class sketch_ref_impl {
  template <class, ::cuda::thread_scope>
  friend class sketch_ref_impl;

 public:
  using key_type      = Key;
  using policy_type   = policy<Key>;
  using register_type = typename policy_type::register_type;
  using cudax_ref     = ::cuda::experimental::cuco::hyperloglog_ref<Key, Scope, policy_type>;

  static constexpr auto thread_scope = Scope;

  template <::cuda::thread_scope NewScope>
  using rebind_scope = sketch_ref_impl<Key, NewScope>;

  __host__ __device__ explicit sketch_ref_impl(::cuda::std::span<::cuda::std::byte> storage)
    : inner_(storage)
  {
  }

  __host__ __device__ explicit sketch_ref_impl(cudax_ref inner) noexcept : inner_(inner) {}

  template <class CooperativeGroup>
  __device__ void clear(CooperativeGroup group) noexcept
  {
    inner_.clear(group);
  }

  __device__ void update(const Key& value) noexcept { inner_.add(value); }

  template <class InputIt>
  __host__ void update(::cuda::stream_ref stream, InputIt first, InputIt last)
  {
    inner_.add(stream, first, last);
  }

  template <class InputIt>
  __host__ void update_async(::cuda::stream_ref stream, InputIt first, InputIt last)
  {
    inner_.add_async(stream, first, last);
  }

  template <class CooperativeGroup, ::cuda::thread_scope OtherScope>
  __device__ void merge(CooperativeGroup group, sketch_ref_impl<Key, OtherScope> other)
  {
    using destination_impl =
      ::cuda::experimental::cuco::__hyperloglog_impl<Key, Scope, policy_type>;
    using source_impl =
      ::cuda::experimental::cuco::__hyperloglog_impl<Key, OtherScope, policy_type>;

    // TODO(NVIDIA/cccl#10211): Use the public cooperative merge once CCCL
    // passes its lightweight source ref by value.
    destination_impl destination{inner_.sketch(), inner_.policy()};
    source_impl source{other.inner_.sketch(), other.inner_.policy()};
    destination.__merge(group, source);
  }

  template <::cuda::thread_scope OtherScope>
  __host__ void merge(::cuda::stream_ref stream, sketch_ref_impl<Key, OtherScope> other)
  {
    inner_.merge(stream, other.inner_);
  }

  template <::cuda::thread_scope OtherScope>
  __host__ void merge_async(::cuda::stream_ref stream, sketch_ref_impl<Key, OtherScope> other)
  {
    inner_.merge_async(stream, other.inner_);
  }

  [[nodiscard]] __device__ ::cuda::std::size_t get_estimate(
    const ::cooperative_groups::thread_block& group) const noexcept
  {
    return inner_.estimate(group);
  }

  template <class HostMemoryResource = ::cuda::mr::legacy_pinned_memory_resource>
  [[nodiscard]] __host__ ::cuda::std::size_t get_estimate(::cuda::stream_ref stream,
                                                          HostMemoryResource host_mr = {}) const
  {
    return inner_.estimate(stream, host_mr);
  }

  [[nodiscard]] __device__ double get_lower_bound(const ::cooperative_groups::thread_block& group,
                                                  std::uint8_t num_std_dev) const noexcept
  {
    const double estimate  = static_cast<double>(get_estimate(group));
    const double non_zero  = static_cast<double>(num_non_zero_registers_(group));
    const double rel_error = relative_error(/*upper_bound=*/false, get_lg_config_k(), num_std_dev);
    const double bound     = estimate / (1.0 + rel_error);
    return bound > non_zero ? bound : non_zero;
  }

  template <class HostMemoryResource = ::cuda::mr::legacy_pinned_memory_resource>
  [[nodiscard]] __host__ double get_lower_bound(::cuda::stream_ref stream,
                                                std::uint8_t num_std_dev,
                                                HostMemoryResource host_mr = {}) const
  {
    ::datasketches::HllUtil<>::checkNumStdDev(num_std_dev);
    const double estimate  = static_cast<double>(get_estimate(stream, host_mr));
    const double non_zero  = static_cast<double>(num_non_zero_registers_(stream, host_mr));
    const double rel_error = relative_error(/*upper_bound=*/false, get_lg_config_k(), num_std_dev);
    const double bound     = estimate / (1.0 + rel_error);
    return bound > non_zero ? bound : non_zero;
  }

  [[nodiscard]] __device__ double get_upper_bound(const ::cooperative_groups::thread_block& group,
                                                  std::uint8_t num_std_dev) const noexcept
  {
    const double estimate  = static_cast<double>(get_estimate(group));
    const double rel_error = relative_error(/*upper_bound=*/true, get_lg_config_k(), num_std_dev);
    return estimate / (1.0 + rel_error);
  }

  template <class HostMemoryResource = ::cuda::mr::legacy_pinned_memory_resource>
  [[nodiscard]] __host__ double get_upper_bound(::cuda::stream_ref stream,
                                                std::uint8_t num_std_dev,
                                                HostMemoryResource host_mr = {}) const
  {
    ::datasketches::HllUtil<>::checkNumStdDev(num_std_dev);
    const double estimate  = static_cast<double>(get_estimate(stream, host_mr));
    const double rel_error = relative_error(/*upper_bound=*/true, get_lg_config_k(), num_std_dev);
    return estimate / (1.0 + rel_error);
  }

  [[nodiscard]] __host__ __device__ std::uint8_t get_lg_config_k() const noexcept
  {
    auto registers = num_registers();
    std::uint8_t lg_k{};
    while (registers > 1) {
      registers >>= 1;
      ++lg_k;
    }
    return lg_k;
  }

  [[nodiscard]] __host__ __device__ ::datasketches::target_hll_type get_target_type() const noexcept
  {
    return ::datasketches::HLL_8;
  }

  [[nodiscard]] __device__ bool is_empty(
    const ::cooperative_groups::thread_block& group) const noexcept
  {
    return num_zero_registers_(group) == num_registers();
  }

  template <class HostMemoryResource = ::cuda::mr::legacy_pinned_memory_resource>
  [[nodiscard]] __host__ bool is_empty(::cuda::stream_ref stream,
                                       HostMemoryResource host_mr = {}) const
  {
    return num_zero_registers_(stream, host_mr) == num_registers();
  }

  [[nodiscard]] __host__ __device__ ::cuda::std::size_t num_registers() const noexcept
  {
    return sketch_bytes() / sizeof(register_type);
  }

  [[nodiscard]] __host__ __device__ ::cuda::std::span<::cuda::std::byte> sketch() const noexcept
  {
    return inner_.sketch();
  }

  [[nodiscard]] __host__ __device__ ::cuda::std::size_t sketch_bytes() const noexcept
  {
    return inner_.sketch_bytes();
  }

  [[nodiscard]] __host__ __device__ static constexpr ::cuda::std::size_t sketch_bytes(
    std::uint8_t lg_k) noexcept
  {
    return cudax_ref::sketch_bytes(typename cudax_ref::precision{static_cast<int>(lg_k)});
  }

  [[nodiscard]] __host__ __device__ static constexpr ::cuda::std::size_t sketch_alignment() noexcept
  {
    return cudax_ref::sketch_alignment();
  }

 private:
  struct zero_count_policy : policy_type {
    [[nodiscard]] __host__ __device__ static ::cuda::std::size_t finalize(double,
                                                                          int num_zeroes,
                                                                          int) noexcept
    {
      return static_cast<::cuda::std::size_t>(num_zeroes);
    }
  };

  using zero_count_ref = ::cuda::experimental::cuco::hyperloglog_ref<Key, Scope, zero_count_policy>;

  [[nodiscard]] __host__ __device__ zero_count_ref zero_count_ref_() const
  {
    return zero_count_ref{inner_.sketch(), zero_count_policy{}};
  }

  [[nodiscard]] __device__ ::cuda::std::size_t num_zero_registers_(
    const ::cooperative_groups::thread_block& group) const noexcept
  {
    return zero_count_ref_().estimate(group);
  }

  template <class HostMemoryResource>
  [[nodiscard]] __host__ ::cuda::std::size_t num_zero_registers_(::cuda::stream_ref stream,
                                                                 HostMemoryResource host_mr) const
  {
    return zero_count_ref_().estimate(stream, host_mr);
  }

  [[nodiscard]] __device__ ::cuda::std::size_t num_non_zero_registers_(
    const ::cooperative_groups::thread_block& group) const noexcept
  {
    return num_registers() - num_zero_registers_(group);
  }

  template <class HostMemoryResource>
  [[nodiscard]] __host__ ::cuda::std::size_t num_non_zero_registers_(
    ::cuda::stream_ref stream, HostMemoryResource host_mr) const
  {
    return num_registers() - num_zero_registers_(stream, host_mr);
  }

  cudax_ref inner_;
};

}  // namespace datasketches::cuda::detail::hll
