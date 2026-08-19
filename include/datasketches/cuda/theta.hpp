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
//! Updates are batch-oriented. A single kernel hashes each key, screens it
//! against theta, filters duplicates, and compacts the survivors; CUB radix
//! sort, unique, and merge primitives then fold those survivors into the
//! retained set. The duplicate filter is best-effort and never affects the
//! result, only how much redundant work reaches the sort. It costs a few percent
//! when duplicates are scattered and pays back several times over when they
//! arrive together, as they do in sorted or grouped input. An update that
//! begins with theta at its maximum is split internally so theta tightens
//! partway through the batch instead of after it. The retained hashes are always
//! ordered and trimmed to the smallest k = 2^lg_k values. Union (merge),
//! intersection, and A-not-B operate directly on those ordered device-resident
//! hashes.
//!
//! The current migration supports primitive device keys, uncompressed ordered
//! compact-v3 serialization, custom seeds, and p-sampling. It does not yet
//! support strings/byte spans, unordered or update-sketch wire images,
//! compressed v4 images, or legacy serialization versions.
//!
//! CUDA work is explicit-resource: construction and every member function that
//! touches the device take a caller-provided `cuda::stream_ref` as the first
//! argument, and construction/deserialization require an explicit memory
//! resource. Accessors that report sketch state take no stream because every
//! operation that changes storage has already synchronized: the retained count
//! determines the next allocation size, so it must be read back on the host.
//! That also means there are no `_async` variants yet, unlike `hll_sketch`.
//! Both follow from the same eager readback and will change together.
//!
//! **Stream lifetime.** The caller MUST keep the stream supplied at construction
//! or deserialization alive until the sketch is destroyed. Retained buffers are
//! rebound to that stream for stream-ordered deallocation.
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
  //! @throws std::invalid_argument if `lg_k` is outside [5, 26] or `p` is
  //!   outside (0, 1].
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
  //! Hashes each key, screens it against theta, and folds the surviving hashes
  //! into the retained set. Large batches are split internally so theta tightens
  //! during the call. Synchronizes `stream` before returning.
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

  //! @brief True iff the sketch has seen no keys.
  //!
  //! @return True iff the sketch has seen no keys.
  [[nodiscard]] bool is_empty() const noexcept;

  //! @brief True iff theta has fallen below its maximum, so the retained count
  //! no longer equals the exact distinct count.
  //!
  //! @return True iff the sketch is in estimation mode.
  [[nodiscard]] bool is_estimation_mode() const noexcept;

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

  //! @brief Current theta as a raw 64-bit hash threshold.
  //!
  //! @return Theta in `[0, 2^63)`.
  [[nodiscard]] std::uint64_t get_theta64() const noexcept;

  //! @brief Current theta as a fraction of the hash space.
  //!
  //! @return Theta in `(0, 1]`.
  [[nodiscard]] double get_theta() const noexcept;

  //! @brief Hash of the seed, used to reject incompatible set operations.
  //!
  //! @return The 16-bit seed hash.
  [[nodiscard]] std::uint16_t get_seed_hash() const noexcept;

  //! @brief Number of hashes the sketch currently retains, at most `2^lg_k`.
  //!
  //! @return The retained entry count.
  [[nodiscard]] std::size_t get_num_retained() const noexcept;

  //! @brief Cardinality estimate.
  //!
  //! @return The cardinality estimate.
  [[nodiscard]] double get_estimate() const noexcept;

  //! @brief Lower bound on the estimate.
  //!
  //! @param[in] num_std_devs Confidence level: 1, 2, or 3.
  //! @return The lower bound, or the exact count when not in estimation mode.
  [[nodiscard]] double get_lower_bound(std::uint8_t num_std_devs) const;

  //! @brief Upper bound on the estimate.
  //!
  //! @param[in] num_std_devs Confidence level: 1, 2, or 3.
  //! @return The upper bound, or the exact count when not in estimation mode.
  [[nodiscard]] double get_upper_bound(std::uint8_t num_std_devs) const;

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
  //! Compact Theta images do not encode nominal k, so `lg_k` must be supplied
  //! by the caller when it differs from the default.
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
