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
#include <cuda/devices>
#include <cuda/std/span>
#include <cuda/stream>
#include <utility>
#include <vector>

#include <cuda/__memory_pool/device_memory_pool.h>

// Upstream datasketches-cpp CPU HLL public header. Declares
// `::datasketches::target_hll_type` and the `HLL_4/6/8` enumerators that the
// using-declarations below pull into our `datasketches::cuda` namespace.
// With our `<datasketches/cuda/...>` include prefix, there is no filename
// collision with this upstream header.
#include <hll.hpp>

#include <datasketches/cuda/detail/hll/hll_sketch_impl.hpp>

namespace datasketches::cuda {

// Bring the upstream target type and enum values into this namespace, matching
// datasketches-cpp's layout where target_hll_type lives alongside hll_sketch.
using target_hll_type = ::datasketches::target_hll_type;
using ::datasketches::HLL_4;
using ::datasketches::HLL_6;
using ::datasketches::HLL_8;

//! @brief Public host class wrapping `cuda::experimental::cuco::hyperloglog`
//! to produce HLL sketches binary-compatible with `datasketches::hll_sketch`.
//!
//! Only `target_hll_type::HLL_8` is supported; other target types throw at
//! construction. The hash, bit-slicing, and Composite estimator match the CPU
//! reference (`datasketches::HllUtil::coupon` + `getCompositeEstimate`).
//!
//! Each sketch is paired with a CUDA stream: either an owned `cuda::stream`
//! that the sketch constructs (when no stream is supplied to the constructor),
//! or a borrowed `cuda::stream_ref` provided by the caller. Methods that touch
//! the device take an optional `stream_ref` argument; when omitted they use
//! the sketch's paired stream.
//!
//! **Stream lifetime when borrowing.** If a constructor (or `deserialize`)
//! overload receives a `stream_ref`, the sketch records the underlying
//! `cudaStream_t` and uses it through its entire lifetime, including its
//! destructor (which issues `cudaFreeAsync` via cudax's internal deleter).
//! The caller MUST keep the supplied stream alive until the sketch is
//! destructed. Destroying the stream first is undefined behavior.
//!
//! **Construction does NOT synchronize.** The zero-fill of the register array
//! is queued on the sketch's paired stream and runs asynchronously, matching
//! the cuco/cudax convention. Methods called on this sketch without an
//! explicit stream argument use the same paired stream and are ordered after
//! the init by construction. Cross-stream usage on a freshly-constructed
//! sketch must synchronize first: e.g. `sketch.stream().sync()` (or wait on
//! it from the foreign stream) before calling
//! `sketch.update(..., other_stream)`.
//!
//! This class is a thin handle. All state and logic live in
//! `detail::hll_sketch_impl`; every method here is a one-line forwarder
//! defined out-of-class in `detail/hll/hll.inl`.
//!
//! @tparam Key The item type the sketch counts. Supported primitive types
//!   match all `datasketches::hll_sketch::update(...)` overloads on the CPU
//!   side: `int8/16/32/64_t`, `uint8/16/32/64_t`, `float`, `double`.
//!   `normalizing_hasher<Key>` applies the same per-type canonicalization
//!   (sign-extension, NaN/-0.0 normalization) before hashing, so GPU and CPU
//!   sketches built from the same inputs produce byte-compatible register arrays.
//!   String / byte-range support requires a separate span-based API (not yet
//!   implemented). Unsupported types produce a compile error.
//! @tparam MR The memory resource type used for device storage. Defaults to
//!   `::cuda::device_memory_pool_ref`.
//! @tparam Scope The thread scope of the underlying atomic operations.
template <class Key,
          class MR                   = ::cuda::device_memory_pool_ref,
          ::cuda::thread_scope Scope = ::cuda::thread_scope_device>
class hll_sketch {
 public:
  using key_type      = Key;
  using policy_type   = typename detail::hll_sketch_impl<Key, MR, Scope>::policy_type;
  using register_type = typename detail::hll_sketch_impl<Key, MR, Scope>::register_type;

  //! @brief Construct a sketch with an owned stream on device 0.
  //!
  //! @param[in] lgK HLL precision parameter (4..18 supported by cudax).
  //! @param[in] tgt Target HLL packing type. Only `HLL_8` is supported.
  //! @param[in] mr Memory resource for device allocations. Defaults to the
  //!   default memory pool of device 0.
  //! @throws std::invalid_argument if `tgt != HLL_8`.
  explicit hll_sketch(std::uint8_t lgK,
                      target_hll_type tgt = HLL_8,
                      MR mr               = ::cuda::device_default_memory_pool(::cuda::devices[0]));

  //! @brief Construct a sketch that borrows the caller's stream.
  //!
  //! @warning The caller must keep `stream` alive until this sketch is
  //!   destructed. The destructor issues async work on it.
  //!
  //! @param[in] lgK HLL precision parameter (4..18 supported by cudax).
  //! @param[in] tgt Target HLL packing type. Only `HLL_8` is supported.
  //! @param[in] mr Memory resource for device allocations.
  //! @param[in] stream Borrowed CUDA stream. Used for init, default-stream
  //!   operations, and destructor async-free.
  //! @throws std::invalid_argument if `tgt != HLL_8`.
  hll_sketch(std::uint8_t lgK, target_hll_type tgt, MR mr, ::cuda::stream_ref stream);

  hll_sketch(const hll_sketch&)            = delete;
  hll_sketch& operator=(const hll_sketch&) = delete;
  hll_sketch(hll_sketch&&)                 = default;
  //! @note Move assignment routes through the impl's custom swap-based
  //! operator to keep the (stream, inner-sketch) pair intact across the
  //! assignment. See `detail::hll_sketch_impl::operator=(&&)`.
  hll_sketch& operator=(hll_sketch&&) = default;
  ~hll_sketch()                       = default;

  //! @brief Reference to the sketch's paired CUDA stream (owned or borrowed).
  [[nodiscard]] ::cuda::stream_ref stream() const noexcept;

  //! @brief Bulk update on the sketch's paired stream.
  template <class InputIt>
  void update(InputIt first, InputIt last);

  //! @brief Bulk update on a caller-provided stream.
  template <class InputIt>
  void update(InputIt first, InputIt last, ::cuda::stream_ref stream);

  //! @brief Bulk update without stream synchronization, on the paired stream.
  template <class InputIt>
  void update_async(InputIt first, InputIt last);

  //! @brief Bulk update without stream synchronization, on a caller stream.
  template <class InputIt>
  void update_async(InputIt first, InputIt last, ::cuda::stream_ref stream);

  //! @brief Composite cardinality estimate on the sketch's paired stream.
  [[nodiscard]] double get_estimate() const;

  //! @brief Composite cardinality estimate on a caller-provided stream.
  [[nodiscard]] double get_estimate(::cuda::stream_ref stream) const;

  //! @brief Lower bound on the estimate at `numStdDev` standard deviations.
  //! Mirrors `HllArray::getLowerBound` (`HllArray-internal.hpp:344-351`) with
  //! `oooFlag=true` (because GPU output forces oooFlag on serialize).
  //! @param[in] numStdDev Confidence level: 1, 2, or 3.
  //! @throws std::invalid_argument if `numStdDev` is outside {1, 2, 3}.
  [[nodiscard]] double get_lower_bound(std::uint8_t numStdDev) const;

  //! @brief Lower bound on the estimate, on a caller-provided stream.
  [[nodiscard]] double get_lower_bound(std::uint8_t numStdDev, ::cuda::stream_ref stream) const;

  //! @brief Upper bound on the estimate at `numStdDev` standard deviations.
  //! Mirrors `HllArray::getUpperBound` (`HllArray-internal.hpp:354-358`).
  //! @param[in] numStdDev Confidence level: 1, 2, or 3.
  //! @throws std::invalid_argument if `numStdDev` is outside {1, 2, 3}.
  [[nodiscard]] double get_upper_bound(std::uint8_t numStdDev) const;

  //! @brief Upper bound on the estimate, on a caller-provided stream.
  [[nodiscard]] double get_upper_bound(std::uint8_t numStdDev, ::cuda::stream_ref stream) const;

  //! @brief Atomic register-wise max merge of `other` into `*this`, on the
  //! paired stream. `other` may have a different memory resource or scope;
  //! the `Key` type must match.
  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge(const hll_sketch<Key, OtherMR, OtherScope>& other);

  //! @brief Merge `other` into `*this` on a caller-provided stream.
  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge(const hll_sketch<Key, OtherMR, OtherScope>& other, ::cuda::stream_ref stream);

  //! @brief Async variant of `merge` on the paired stream.
  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge_async(const hll_sketch<Key, OtherMR, OtherScope>& other);

  //! @brief Async variant of `merge` on a caller-provided stream.
  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge_async(const hll_sketch<Key, OtherMR, OtherScope>& other, ::cuda::stream_ref stream);

  //! @brief HLL precision parameter the sketch was constructed with.
  [[nodiscard]] std::uint8_t get_lg_config_k() const noexcept;

  //! @brief Target HLL packing type the sketch was constructed with.
  [[nodiscard]] target_hll_type get_target_type() const noexcept;

  //! @brief True iff every register is zero. Synchronizes the paired stream.
  [[nodiscard]] bool is_empty() const;

  //! @brief True iff every register is zero. Synchronizes the given stream.
  [[nodiscard]] bool is_empty(::cuda::stream_ref stream) const;

  //! @brief Serialize to the Datasketches compact wire format. For HLL_8 this
  //! is byte-identical to `serialize_updatable`.
  //!
  //! GPU-produced blobs always set `oooFlag=1`, write `hipAccum=0`, and write
  //! `cur_min=0` (the HLL_8 invariant). All other bytes (lgK, kxq0, kxq1,
  //! num_at_cur_min, register array) are deterministic functions of the device
  //! register state and round-trip exactly.
  [[nodiscard]] std::vector<std::uint8_t> serialize_compact() const;

  //! @brief Serialize to the Datasketches compact wire format, on a
  //! caller-provided stream.
  [[nodiscard]] std::vector<std::uint8_t> serialize_compact(::cuda::stream_ref stream) const;

  //! @brief Serialize to the Datasketches updatable wire format. For HLL_8 the
  //! compact and updatable forms are byte-identical except the `compact_flag`
  //! bit in the FLAGS byte.
  [[nodiscard]] std::vector<std::uint8_t> serialize_updatable() const;

  //! @brief Serialize to the Datasketches updatable wire format, on a
  //! caller-provided stream.
  [[nodiscard]] std::vector<std::uint8_t> serialize_updatable(::cuda::stream_ref stream) const;

  //! @brief Deserialize an HLL_8 blob into a sketch with an owned stream on
  //! device 0.
  //!
  //! The trailing fields (`kxq0`, `kxq1`, `numAtCurMin`, `hipAccum`) in the
  //! preamble are NOT propagated into the GPU instance; they will be
  //! recomputed by `reduce_hll8` on next serialize/estimate. This is safe
  //! because they are deterministic functions of the register array.
  //!
  //! @param[in] bytes Wire-format blob: 40-byte preamble + `2^lgK` register bytes.
  //! @param[in] mr Memory resource for device allocations. Defaults to the
  //!   default memory pool of device 0.
  //! @throws std::invalid_argument if `bytes` is too short, mismatched in size,
  //!   not in HLL mode, or not HLL_8 target.
  static hll_sketch deserialize(::cuda::std::span<const std::uint8_t> bytes,
                                MR mr = ::cuda::device_default_memory_pool(::cuda::devices[0]));

  //! @brief Deserialize an HLL_8 blob into a sketch that borrows the caller's
  //! stream.
  //!
  //! @warning The caller must keep `stream` alive until the returned sketch is
  //!   destructed.
  //!
  //! @param[in] bytes Wire-format blob: 40-byte preamble + `2^lgK` register bytes.
  //! @param[in] mr Memory resource for device allocations.
  //! @param[in] stream Borrowed CUDA stream.
  //! @throws std::invalid_argument if `bytes` is malformed.
  static hll_sketch deserialize(::cuda::std::span<const std::uint8_t> bytes,
                                MR mr,
                                ::cuda::stream_ref stream);

  //! @brief Number of HLL registers (`2^lgK`).
  [[nodiscard]] std::size_t num_registers() const noexcept;

 private:
  template <class K_, class M_, ::cuda::thread_scope S_>
  friend class hll_sketch;

  detail::hll_sketch_impl<Key, MR, Scope> impl_;
};

}  // namespace datasketches::cuda

#include <datasketches/cuda/detail/hll/hll.inl>
