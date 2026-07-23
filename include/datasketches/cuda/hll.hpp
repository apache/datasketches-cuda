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
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>
#include <utility>
#include <vector>

#include <hll.hpp>

#include <datasketches/cuda/detail/hll/sketch_impl.hpp>
#include <datasketches/cuda/hll_ref.cuh>

namespace datasketches::cuda {

using target_hll_type = ::datasketches::target_hll_type;
using ::datasketches::HLL_4;
using ::datasketches::HLL_6;
using ::datasketches::HLL_8;

//! @brief Public host class wrapping `cuda::experimental::cuco::hyperloglog`
//! to produce HLL sketches binary-compatible with `datasketches::hll_sketch`.
//!
//! Only `target_hll_type::HLL_8` is supported; other target types throw at
//! construction.
//!
//! CUDA work is explicit-resource: construction and every member function that
//! touches the device take a caller-provided `cuda::stream_ref` as the first
//! argument, and construction/deserialization require an explicit memory
//! resource. Host-returning methods synchronize the supplied stream before
//! returning.
//!
//! **Stream lifetime.** The caller MUST keep the stream supplied at construction
//! or deserialization alive until the sketch is destroyed. The backing cudax
//! object may issue async deallocation work on that construction stream. The
//! caller must also ensure any streams used with `update_async` or `merge_async`
//! have completed, or are otherwise ordered before the construction stream, before
//! destroying the sketch.
//!
//! @tparam Key The item type the sketch counts. Supported primitive types
//!   `int8/16/32/64_t`, `uint8/16/32/64_t`, `float`, `double`.
//! @tparam MR The memory resource type used for device storage. Defaults to
//!   `::cuda::device_memory_pool_ref`.
//! @tparam Scope The thread scope of the underlying atomic operations.
template <class Key,
          class MR                   = ::cuda::device_memory_pool_ref,
          ::cuda::thread_scope Scope = ::cuda::thread_scope_device>
class hll_sketch {
 public:
  using key_type      = Key;
  using policy_type   = typename detail::hll::sketch_impl<Key, MR, Scope>::policy_type;
  using register_type = typename detail::hll::sketch_impl<Key, MR, Scope>::register_type;
  using ref_type      = hll_sketch_ref<Key, Scope>;

  //! @brief Construct a sketch on a caller-provided stream.
  //!
  //! @param[in] stream CUDA stream used for stream-ordered initialization.
  //! @param[in] mr Memory resource for device allocations.
  //! @param[in] lgK HLL precision parameter (4..18).
  //! @param[in] tgt Target HLL packing type.
  hll_sketch(::cuda::stream_ref stream, MR mr, std::uint8_t lgK, target_hll_type tgt = HLL_8);

  hll_sketch(const hll_sketch&)            = delete;
  hll_sketch& operator=(const hll_sketch&) = delete;
  hll_sketch(hll_sketch&&)                 = default;
  hll_sketch& operator=(hll_sketch&&)      = default;
  ~hll_sketch()                            = default;

  //! @brief Bulk update on a caller-provided stream.
  //!
  //! @tparam InputIt Iterator type.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] first Iterator to the first element to update.
  //! @param[in] last Iterator to the last element to update.
  template <class InputIt>
  void update(::cuda::stream_ref stream, InputIt first, InputIt last);

  //! @brief Bulk update without stream synchronization.
  //!
  //! @tparam InputIt Iterator type.
  //!
  //! @param[in] stream CUDA stream this operation is enqueued in.
  //! @param[in] first Iterator to the first element to update.
  //! @param[in] last Iterator to the last element to update.
  //!
  //! @warning The caller must synchronize or order `stream` before destroying
  //!   this sketch.
  template <class InputIt>
  void update_async(::cuda::stream_ref stream, InputIt first, InputIt last);

  //! @brief Return a non-owning mutable ref to the device register storage.
  //!
  //! The returned ref must not outlive this sketch. The caller remains
  //! responsible for stream ordering and for choosing a valid thread scope.
  [[nodiscard]] ref_type ref() noexcept;

  //! @brief Cardinality estimate. Synchronizes `stream` before returning.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return The cardinality estimate.
  //!
  //! @note The current CCCL estimate contract returns `size_t`, so this
  //!   `double` is temporarily integer-valued.
  //!
  //! @todo NVIDIA/cccl#10209: Preserve the Composite estimator's fractional
  //!   result once CCCL supports a policy-defined estimate result type.
  [[nodiscard]] double get_estimate(::cuda::stream_ref stream) const;

  //! @brief Lower bound on the estimate. Synchronizes `stream` before returning.
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] numStdDev Confidence level: 1, 2, or 3.
  //! @throws std::invalid_argument if `numStdDev` is outside {1, 2, 3}.
  [[nodiscard]] double get_lower_bound(::cuda::stream_ref stream, std::uint8_t numStdDev) const;

  //! @brief Upper bound on the estimate. Synchronizes `stream` before returning.
  //! Mirrors `HllArray::getUpperBound` (`HllArray-internal.hpp:354-358`).
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] numStdDev Confidence level: 1, 2, or 3.
  //! @throws std::invalid_argument if `numStdDev` is outside {1, 2, 3}.
  [[nodiscard]] double get_upper_bound(::cuda::stream_ref stream, std::uint8_t numStdDev) const;

  //! @brief Merge `other` into `*this` on a caller-provided stream.
  //!
  //! @tparam OtherMR Memory resource type of `other`.
  //! @tparam OtherScope Thread scope of `other`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] other The other sketch to merge into `*this`.
  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge(::cuda::stream_ref stream, const hll_sketch<Key, OtherMR, OtherScope>& other);

  //! @brief Async variant of `merge`.
  //!
  //! @tparam OtherMR Memory resource type of `other`.
  //! @tparam OtherScope Thread scope of `other`.
  //!
  //! @param[in] stream CUDA stream this operation is enqueued in.
  //! @param[in] other The other sketch to merge into `*this`.
  //!
  //! @warning The caller must synchronize or order `stream` before destroying
  //!   either sketch.
  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge_async(::cuda::stream_ref stream, const hll_sketch<Key, OtherMR, OtherScope>& other);

  //! @brief HLL precision parameter the sketch was constructed with.
  //!
  //! @return The HLL precision parameter.
  [[nodiscard]] std::uint8_t get_lg_config_k() const noexcept;

  //! @brief Target HLL packing type the sketch was constructed with.
  //!
  //! @return The target HLL packing type.
  [[nodiscard]] target_hll_type get_target_type() const noexcept;

  //! @brief True iff every register is zero. Synchronizes `stream` before returning.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return True iff every register is zero.
  [[nodiscard]] bool is_empty(::cuda::stream_ref stream) const;

  //! @brief Serialize to the Datasketches compact wire format. For HLL_8 this
  //! is byte-identical to `serialize_updatable`.
  //!
  //! GPU-produced blobs always set `oooFlag=1`, write `hipAccum=0`, and write
  //! `cur_min=0` (the HLL_8 invariant). All other bytes (lgK, kxq0, kxq1,
  //! num_at_cur_min, register array) are deterministic functions of the device
  //! register state and round-trip exactly.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return The serialized sketch.
  [[nodiscard]] std::vector<std::uint8_t> serialize_compact(::cuda::stream_ref stream) const;

  //! @brief Serialize to the Datasketches updatable wire format. For HLL_8 the
  //! compact and updatable forms are byte-identical except the `compact_flag`
  //! bit in the FLAGS byte.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return The serialized sketch.
  [[nodiscard]] std::vector<std::uint8_t> serialize_updatable(::cuda::stream_ref stream) const;

  //! @brief Deserialize an HLL_8 blob into a sketch.
  //!
  //! The trailing fields (`kxq0`, `kxq1`, `numAtCurMin`, `hipAccum`) in the
  //! preamble are NOT propagated into the GPU instance; they will be
  //! recomputed by `reduce_hll8` on next serialize/estimate. This is safe
  //! because they are deterministic functions of the register array.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] bytes Wire-format blob: 40-byte preamble + `2^lgK` register bytes.
  //! @param[in] mr Memory resource for device allocations.
  //! @throws std::invalid_argument if `bytes` is too short, mismatched in size,
  //!   not in HLL mode, or not HLL_8 target.
  //! @return The deserialized sketch.
  static hll_sketch deserialize(::cuda::stream_ref stream,
                                ::cuda::std::span<const std::uint8_t> bytes,
                                MR mr);

  //! @brief Number of HLL registers (`2^lgK`).
  //!
  //! @return The number of HLL registers.
  [[nodiscard]] std::size_t num_registers() const noexcept;

 private:
  template <class K_, class M_, ::cuda::thread_scope S_>
  friend class hll_sketch;  // Allow the implementation details to access the public API.

  detail::hll::sketch_impl<Key, MR, Scope> impl_;  // Implementation details.
};

}  // namespace datasketches::cuda

#include <datasketches/cuda/detail/hll/hll.inl>
