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

#include <hll.hpp>

#include <datasketches/cuda/detail/hll/sketch_ref_impl.cuh>

#include <cooperative_groups.h>

namespace datasketches::cuda {

template <class Key, class MR, ::cuda::thread_scope Scope>
class hll_sketch;

//! @brief Non-owning device reference to caller-managed HLL_8 register storage.
//!
//! `hll_sketch_ref` applies the same DataSketches key normalization, MurmurHash3
//! hashing, register selection, and register update policy as `hll_sketch`
//! without allocating or owning storage. The storage layout is the CCCL HLL
//! layout: one `register_type` value per register and `2^lgK` registers.
//!
//! Construction does not initialize storage. Call `clear(group)` before first
//! use unless the span already contains a valid sketch. The referenced storage
//! must remain alive and accessible for every operation using the ref.
//!
//! All threads in the supplied cooperative group must participate in
//! cooperative operations. `clear` and `merge` do not add a trailing group
//! synchronization; callers must synchronize before a dependent operation.
//!
//! `thread_scope_device` is required when multiple thread blocks may update the
//! same sketch. `thread_scope_block` is valid only while one block exclusively
//! owns all accesses to the storage.
//!
//! @tparam Key The item type the sketch counts. Supported primitive types are
//!   `int8/16/32/64_t`, `uint8/16/32/64_t`, `float`, and `double`.
//! @tparam Scope The CUDA thread scope used for atomic register updates.
template <class Key, ::cuda::thread_scope Scope = ::cuda::thread_scope_device>
class hll_sketch_ref {
  using impl_type = detail::hll::sketch_ref_impl<Key, Scope>;

  template <class, ::cuda::thread_scope>
  friend class hll_sketch_ref;

  template <class, class, ::cuda::thread_scope>
  friend class hll_sketch;

 public:
  //! @brief Item type counted by this sketch.
  using key_type = typename impl_type::key_type;

  //! @brief DataSketches-compatible hash and register-update policy.
  using policy_type = typename impl_type::policy_type;

  //! @brief Register storage type used by the underlying CCCL HLL.
  using register_type = typename impl_type::register_type;

  //! @brief CUDA thread scope used for atomic register updates.
  static constexpr auto thread_scope = impl_type::thread_scope;

  //! @brief Rebind this ref type to a different CUDA thread scope.
  //!
  //! Rebinding does not change the storage. The caller must ensure every access
  //! satisfies the synchronization and ownership guarantees of `NewScope`.
  //!
  //! @tparam NewScope CUDA thread scope for the rebound ref type.
  template <::cuda::thread_scope NewScope>
  using rebind_scope = hll_sketch_ref<Key, NewScope>;

  //! @brief Construct a ref over caller-managed HLL register storage.
  //!
  //! The span must contain exactly `sketch_bytes(lgK)` bytes for one supported
  //! precision and begin at an address aligned to `sketch_alignment()`.
  //! Construction does not clear or otherwise modify the registers.
  //!
  //! @param[in] storage Mutable storage containing the HLL registers.
  //!
  //! @throws std::invalid_argument On the host if CCCL rejects the storage
  //!   alignment or inferred precision. Device-side CCCL contract violations
  //!   terminate kernel execution.
  __host__ __device__ explicit hll_sketch_ref(::cuda::std::span<::cuda::std::byte> storage)
    : impl_(storage)
  {
  }

  //! @brief Cooperatively clear all registers.
  //!
  //! Every thread in `group` must call this function with the same ref. The
  //! caller must synchronize `group` before any thread performs a dependent
  //! update, merge, or estimate.
  //!
  //! @tparam CooperativeGroup CUDA cooperative group type.
  //! @param[in] group Cooperative group partitioning the register array.
  template <class CooperativeGroup>
  __device__ void clear(CooperativeGroup group) noexcept
  {
    impl_.clear(group);
  }

  //! @brief Update one register from the calling thread.
  //!
  //! @param[in] value Item to add to the sketch.
  __device__ void update(const Key& value) noexcept { impl_.update(value); }

  //! @brief Cooperatively merge `other` into this sketch.
  //!
  //! Every thread in `group` must call this function with the same refs.
  //! `other` is passed by value as a lightweight non-owning handle and its
  //! storage is not modified. The caller must synchronize `group` before a
  //! dependent operation observes the destination.
  //!
  //! @tparam CooperativeGroup CUDA cooperative group type.
  //! @tparam OtherScope CUDA thread scope of the source ref.
  //! @param[in] group Cooperative group partitioning the register arrays.
  //! @param[in] other Source sketch merged into `*this`.
  //! @pre `sketch_bytes() == other.sketch_bytes()`.
  template <class CooperativeGroup, ::cuda::thread_scope OtherScope>
  __device__ void merge(CooperativeGroup group, hll_sketch_ref<Key, OtherScope> other)
  {
    impl_.merge(group, other.impl_);
  }

  //! @brief Cooperatively compute a truncated Composite estimate.
  //!
  //! @param[in] group Thread block reducing the register array.
  //! @return DataSketches Composite estimate converted to `size_t`.
  //! @todo NVIDIA/cccl#10209: Return `double` once CCCL supports a
  //!   policy-defined estimate result type.
  [[nodiscard]] __device__ ::cuda::std::size_t get_estimate(
    const ::cooperative_groups::thread_block& group) const noexcept
  {
    return impl_.get_estimate(group);
  }

  //! @brief Compute a truncated Composite estimate on a host stream.
  //!
  //! This function synchronizes `stream` before returning.
  //!
  //! @tparam HostMemoryResource Host memory resource used for the temporary
  //!   register copy.
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] host_mr Host memory resource used for the temporary copy.
  //! @return DataSketches Composite estimate converted to `size_t`.
  //! @todo NVIDIA/cccl#10209: Return `double` once CCCL supports a
  //!   policy-defined estimate result type.
  template <class HostMemoryResource = ::cuda::mr::legacy_pinned_memory_resource>
  [[nodiscard]] __host__ ::cuda::std::size_t get_estimate(::cuda::stream_ref stream,
                                                          HostMemoryResource host_mr = {}) const
  {
    return impl_.get_estimate(stream, host_mr);
  }

  //! @brief Cooperatively compute the lower confidence bound.
  //!
  //! @param[in] group Thread block reducing the register array.
  //! @param[in] num_std_dev Confidence level. Must be 1, 2, or 3.
  //! @return Lower confidence bound on the cardinality estimate.
  [[nodiscard]] __device__ double get_lower_bound(const ::cooperative_groups::thread_block& group,
                                                  std::uint8_t num_std_dev) const noexcept
  {
    return impl_.get_lower_bound(group, num_std_dev);
  }

  //! @brief Compute the lower confidence bound on a host stream.
  //!
  //! @tparam HostMemoryResource Host memory resource used for temporary
  //!   register copies.
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] num_std_dev Confidence level: 1, 2, or 3.
  //! @param[in] host_mr Host memory resource used for temporary copies.
  //! @throws std::invalid_argument if `num_std_dev` is outside `[1, 3]`.
  //! @return Lower confidence bound on the cardinality estimate.
  template <class HostMemoryResource = ::cuda::mr::legacy_pinned_memory_resource>
  [[nodiscard]] __host__ double get_lower_bound(::cuda::stream_ref stream,
                                                std::uint8_t num_std_dev,
                                                HostMemoryResource host_mr = {}) const
  {
    return impl_.get_lower_bound(stream, num_std_dev, host_mr);
  }

  //! @brief Cooperatively compute the upper confidence bound.
  //!
  //! @param[in] group Thread block reducing the register array.
  //! @param[in] num_std_dev Confidence level. Must be 1, 2, or 3.
  //! @return Upper confidence bound on the cardinality estimate.
  [[nodiscard]] __device__ double get_upper_bound(const ::cooperative_groups::thread_block& group,
                                                  std::uint8_t num_std_dev) const noexcept
  {
    return impl_.get_upper_bound(group, num_std_dev);
  }

  //! @brief Compute the upper confidence bound on a host stream.
  //!
  //! @tparam HostMemoryResource Host memory resource used for the temporary
  //!   register copy.
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] num_std_dev Confidence level: 1, 2, or 3.
  //! @param[in] host_mr Host memory resource used for the temporary copy.
  //! @throws std::invalid_argument if `num_std_dev` is outside `[1, 3]`.
  //! @return Upper confidence bound on the cardinality estimate.
  template <class HostMemoryResource = ::cuda::mr::legacy_pinned_memory_resource>
  [[nodiscard]] __host__ double get_upper_bound(::cuda::stream_ref stream,
                                                std::uint8_t num_std_dev,
                                                HostMemoryResource host_mr = {}) const
  {
    return impl_.get_upper_bound(stream, num_std_dev, host_mr);
  }

  //! @brief Return the HLL precision inferred from the register storage.
  //! @return HLL precision parameter in `[4, 18]`.
  [[nodiscard]] __host__ __device__ std::uint8_t get_lg_config_k() const noexcept
  {
    return impl_.get_lg_config_k();
  }

  //! @brief Return the target HLL packing type.
  //! @return `target_hll_type::HLL_8`.
  [[nodiscard]] __host__ __device__ ::datasketches::target_hll_type get_target_type() const noexcept
  {
    return impl_.get_target_type();
  }

  //! @brief Cooperatively test whether every register is zero.
  //! @param[in] group Thread block reducing the register array.
  //! @return True iff every register is zero.
  [[nodiscard]] __device__ bool is_empty(
    const ::cooperative_groups::thread_block& group) const noexcept
  {
    return impl_.is_empty(group);
  }

  //! @brief Test whether every register is zero on a host stream.
  //!
  //! @tparam HostMemoryResource Host memory resource used for the temporary
  //!   register copy.
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] host_mr Host memory resource used for the temporary copy.
  //! @return True iff every register is zero.
  template <class HostMemoryResource = ::cuda::mr::legacy_pinned_memory_resource>
  [[nodiscard]] __host__ bool is_empty(::cuda::stream_ref stream,
                                       HostMemoryResource host_mr = {}) const
  {
    return impl_.is_empty(stream, host_mr);
  }

  //! @brief Return the number of HLL registers.
  //! @return Number of registers (`2^lgK`).
  [[nodiscard]] __host__ __device__ ::cuda::std::size_t num_registers() const noexcept
  {
    return impl_.num_registers();
  }

  //! @brief Return the referenced register storage.
  //! @return Mutable byte span over the caller-owned storage.
  [[nodiscard]] __host__ __device__ ::cuda::std::span<::cuda::std::byte> sketch() const noexcept
  {
    return impl_.sketch();
  }

  //! @brief Return the referenced storage size in bytes.
  //! @return Number of bytes used by the register array.
  [[nodiscard]] __host__ __device__ ::cuda::std::size_t sketch_bytes() const noexcept
  {
    return impl_.sketch_bytes();
  }

  //! @brief Return the exact storage size required for `lgK`.
  //! @param[in] lg_k HLL precision parameter in `[4, 18]`.
  //! @return Number of caller-managed bytes required for the register array.
  [[nodiscard]] __host__ __device__ static constexpr ::cuda::std::size_t sketch_bytes(
    std::uint8_t lg_k) noexcept
  {
    return impl_type::sketch_bytes(lg_k);
  }

  //! @brief Return the alignment required for caller-managed storage.
  //! @return Required starting-address alignment in bytes.
  [[nodiscard]] __host__ __device__ static constexpr ::cuda::std::size_t sketch_alignment() noexcept
  {
    return impl_type::sketch_alignment();
  }

 private:
  __host__ __device__ explicit hll_sketch_ref(impl_type impl) noexcept : impl_(impl) {}

  impl_type impl_;
};

}  // namespace datasketches::cuda
