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
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>
#include <utility>
#include <vector>

#include <datasketches/cuda/detail/theta/policy.cuh>
#include <datasketches/cuda/detail/theta/sketch_impl.hpp>

namespace datasketches::cuda {

//! @brief GPU Theta sketch with ordered compact serialization compatible with
//! the datasketches::compact_theta_sketch serialization version 3.
//!
//! Updates use an occupancy-sized persistent grid. Each block uses the device's
//! available opt-in shared memory for an exact set of its partition's smallest
//! hashes and emits at most k + 1 candidates. One device-wide sort and unique
//! operation then produces the ordered retained set and theta.
//!
//! The current migration supports primitive device keys, uncompressed ordered
//! compact-v3 serialization, custom seeds, and p-sampling. It does not yet
//! support strings/byte spans, unordered or update-sketch wire images,
//! compressed v4 images, or legacy serialization versions.
//!
//! CUDA work is explicit-resource: construction and every member function that
//! touches device state takes a caller-provided `cuda::stream_ref`. Host-returning
//! methods synchronize that stream before returning.
//!
//! **Stream lifetime.** The caller MUST keep the stream supplied at construction
//! or deserialization alive until the sketch is destroyed. Retained buffers are
//! bound to that stream for stream-ordered deallocation. The caller must also
//! ensure any stream used with `update_async` has completed, or is otherwise
//! ordered before the construction stream, before destroying the sketch.
//!
//! @tparam Key Primitive input key type.
//! @tparam MR Device-accessible memory resource type. Defaults to
//!   `::cuda::device_memory_pool_ref`.
template <class Key, class MR = ::cuda::device_memory_pool_ref>
class theta_sketch {
 public:
  using key_type  = Key;
  using hash_type = std::uint64_t;

  static constexpr std::uint8_t default_lg_k  = detail::theta::default_lg_k;
  static constexpr std::uint64_t default_seed = detail::theta::default_seed;

  //! @brief Construct an empty sketch on a caller-provided stream.
  //!
  //! @param[in] stream CUDA stream used for stream-ordered initialization.
  //! @param[in] mr Memory resource for device allocations.
  //! @param[in] lg_k Base 2 logarithm of the nominal number of retained entries.
  //! @param[in] seed Hash seed; sketches built with different seeds cannot be
  //!   combined.
  //! @param[in] p Sampling probability, in (0, 1].
  //! @throws std::invalid_argument if `lg_k` is not 12 or `p` is outside (0, 1].
  theta_sketch(::cuda::stream_ref stream,
               MR mr,
               std::uint8_t lg_k  = default_lg_k,
               std::uint64_t seed = default_seed,
               float p            = 1.0F);

  theta_sketch(const theta_sketch&)            = delete;
  theta_sketch& operator=(const theta_sketch&) = delete;
  theta_sketch(theta_sketch&&)                 = default;
  theta_sketch& operator=(theta_sketch&&)      = default;
  ~theta_sketch()                              = default;

  //! @brief Bulk update on a caller-provided stream.
  //!
  //! Hashes each key, performs block-local reduction, and folds the surviving
  //! hashes into the retained set. Synchronizes `stream` before returning.
  //!
  //! @tparam RandomAccessIt Random-access iterator type over device-accessible
  //!   keys.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] first Iterator to the first element to update.
  //! @param[in] last Iterator past the last element to update.
  //! @throws std::invalid_argument if `last` precedes `first`.
  template <class RandomAccessIt>
  void update(::cuda::stream_ref stream, RandomAccessIt first, RandomAccessIt last);

  //! @brief Bulk update without stream synchronization.
  //!
  //! Operations on the same sketch must be ordered on the same stream or by the
  //! caller. The caller must also ensure the stream has completed before
  //! destroying the sketch.
  template <class RandomAccessIt>
  void update_async(::cuda::stream_ref stream, RandomAccessIt first, RandomAccessIt last);

  //! @brief Replace this sketch with the union of this and `other`.
  //!
  //! Synchronizes `stream` before returning.
  //!
  //! @tparam OtherMR Memory resource type of `other`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] other The other sketch to union into `*this`.
  //! @throws std::invalid_argument if the two sketches disagree on seed hash.
  template <class OtherMR>
  void merge(::cuda::stream_ref stream, const theta_sketch<Key, OtherMR>& other);

  //! @brief Replace this sketch with the intersection of this and `other`.
  //!
  //! Synchronizes `stream` before returning.
  //!
  //! @tparam OtherMR Memory resource type of `other`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] other The other sketch to intersect with `*this`.
  //! @throws std::invalid_argument if the two sketches disagree on seed hash.
  template <class OtherMR>
  void intersect(::cuda::stream_ref stream, const theta_sketch<Key, OtherMR>& other);

  //! @brief Replace this sketch with the set difference this-minus-`other`.
  //!
  //! Synchronizes `stream` before returning.
  //!
  //! @tparam OtherMR Memory resource type of `other`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] other The sketch to subtract from `*this`.
  //! @throws std::invalid_argument if the two sketches disagree on seed hash.
  template <class OtherMR>
  void a_not_b(::cuda::stream_ref stream, const theta_sketch<Key, OtherMR>& other);

  //! @brief Restore the initial empty state.
  //!
  //! Synchronizes `stream` before returning.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  void reset(::cuda::stream_ref stream);

  //! @brief True iff the sketch has seen no keys. Synchronizes `stream`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return True iff the sketch has seen no keys.
  [[nodiscard]] bool is_empty(::cuda::stream_ref stream) const;

  //! @brief True iff theta has fallen below its maximum, so the retained count
  //! no longer equals the exact distinct count. Synchronizes `stream`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return True iff the sketch is in estimation mode.
  [[nodiscard]] bool is_estimation_mode(::cuda::stream_ref stream) const;

  //! @brief True iff the retained hashes are in ascending order.
  //!
  //! Always true: retained hashes are kept ordered so the compact image can be
  //! written without a sort.
  //!
  //! @return True.
  [[nodiscard]] bool is_ordered() const noexcept;

  //! @brief Base 2 logarithm of the nominal number of entries.
  //!
  //! @return The `lg_k` the sketch was constructed with.
  [[nodiscard]] std::uint8_t get_lg_k() const noexcept;

  //! @brief Current theta as a raw 64-bit hash threshold. Synchronizes `stream`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return Theta in `[0, 2^63)`.
  [[nodiscard]] std::uint64_t get_theta64(::cuda::stream_ref stream) const;

  //! @brief Current theta as a fraction of the hash space. Synchronizes `stream`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return Theta in `(0, 1]`.
  [[nodiscard]] double get_theta(::cuda::stream_ref stream) const;

  //! @brief Hash of the seed, used to reject incompatible set operations.
  //!
  //! @return The 16-bit seed hash.
  [[nodiscard]] std::uint16_t get_seed_hash() const noexcept;

  //! @brief Number of retained hashes. Synchronizes `stream`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return The retained entry count.
  [[nodiscard]] std::size_t get_num_retained(::cuda::stream_ref stream) const;

  //! @brief Cardinality estimate. Synchronizes `stream`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return The cardinality estimate.
  [[nodiscard]] double get_estimate(::cuda::stream_ref stream) const;

  //! @brief Lower bound on the estimate. Synchronizes `stream`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] num_std_devs Confidence level: 1, 2, or 3.
  //! @return The lower bound, or the exact count when not in estimation mode.
  [[nodiscard]] double get_lower_bound(::cuda::stream_ref stream, std::uint8_t num_std_devs) const;

  //! @brief Upper bound on the estimate. Synchronizes `stream`.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] num_std_devs Confidence level: 1, 2, or 3.
  //! @return The upper bound, or the exact count when not in estimation mode.
  [[nodiscard]] double get_upper_bound(::cuda::stream_ref stream, std::uint8_t num_std_devs) const;

  //! @brief Copy the ordered retained hashes to host memory.
  //!
  //! Synchronizes `stream` before returning.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return The retained hashes, in ascending order.
  [[nodiscard]] std::vector<hash_type> get_retained_hashes(::cuda::stream_ref stream) const;

  //! @brief Serialize as an ordered, uncompressed compact Theta v3 image.
  //!
  //! Synchronizes `stream` before returning.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @return The serialized sketch.
  [[nodiscard]] std::vector<std::uint8_t> serialize_compact(::cuda::stream_ref stream) const;

  //! @brief Deserialize an ordered, uncompressed compact Theta v3 image.
  //!
  //! This prototype accepts only the default `lg_k` of 12.
  //!
  //! @param[in] stream CUDA stream this operation is executed in.
  //! @param[in] bytes Wire-format compact Theta v3 image.
  //! @param[in] mr Memory resource for device allocations.
  //! @param[in] lg_k Base 2 logarithm of the nominal number of retained entries.
  //! @param[in] seed Hash seed the image was produced with.
  //! @throws std::invalid_argument if `bytes` is malformed, is not an ordered
  //!   uncompressed v3 image, disagrees with `seed`, or holds more entries than
  //!   `lg_k` allows.
  //! @return The deserialized sketch.
  static theta_sketch deserialize(::cuda::stream_ref stream,
                                  ::cuda::std::span<const std::uint8_t> bytes,
                                  MR mr,
                                  std::uint8_t lg_k  = default_lg_k,
                                  std::uint64_t seed = default_seed);

 private:
  template <class, class>
  friend class theta_sketch;  // Allow the implementation details to access the public API.

  detail::theta::sketch_impl<Key, MR> impl_;  // Implementation details.
};

}  // namespace datasketches::cuda

#include <datasketches/cuda/detail/theta/theta.inl>
