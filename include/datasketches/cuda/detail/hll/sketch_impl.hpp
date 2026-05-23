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

#include <algorithm>
#include <cstdint>
#include <cuda/devices>
#include <cuda/std/span>
#include <cuda/stream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <cuda_runtime.h>

#include <cuda/__memory_pool/device_memory_pool.h>

#include <cuda/experimental/__cuco/hyperloglog.cuh>

#include <hll.hpp>

#include <datasketches/cuda/detail/common/error.hpp>
#include <datasketches/cuda/detail/hll/composite_finalizer.hpp>
#include <datasketches/cuda/detail/hll/policy.cuh>
#include <datasketches/cuda/detail/hll/preamble.hpp>
#include <datasketches/cuda/detail/hll/reduction_state.hpp>

namespace datasketches::cuda::detail::hll {

// Implementation behind the public `datasketches::cuda::hll_sketch` handle.
// Owns the cudax HyperLogLog sketch, the stream paired with it (either an
// owned `cuda::stream` constructed for this sketch, or a borrowed
// `cuda::stream_ref` supplied by the caller), and the precision / target
// metadata. All operations live here; the handle is a thin forwarder.
//
// Declared as a `struct` because everything is intended for use by the handle
// (or by other impl methods); there is no user-facing API surface to protect
// with access control. Pattern mirrors cudax's `__hyperloglog_impl` behind
// `cuda::experimental::cuco::hyperloglog`.
//
// Stream lifetime:
//   - If constructed with a caller-supplied `stream_ref`, the sketch borrows
//     it: the caller MUST keep that stream alive for the sketch's lifetime.
//     The destructor issues `cudaFreeAsync` on the borrowed stream via
//     cudax's internal deleter; a stream destroyed before the sketch is UB.
//   - If constructed without a stream, the sketch owns a `cuda::stream` on
//     device 0. Member declaration order ensures the owned stream outlives
//     `inner_`.
template <class Key,
          class MR                   = ::cuda::device_memory_pool_ref,
          ::cuda::thread_scope Scope = ::cuda::thread_scope_device>
struct sketch_impl {
  using key_type      = Key;
  using policy_type   = policy<Key>;
  using register_type = typename policy_type::register_type;

  using cudax_hll = ::cuda::experimental::cuco::hyperloglog<Key, MR, Scope, policy_type>;
  using precision = typename cudax_hll::precision;

  // Member declaration order matters: stream_ must precede inner_ so that
  // for the owned-stream case, the stream destructs AFTER inner_ has released
  // its device buffer (which was allocated on that stream). For the borrowed
  // case, lifetime is on the caller; cudax internally holds the underlying
  // cudaStream_t handle from the stream_ref and uses it for cudaFreeAsync.
  std::variant<::cuda::stream, ::cuda::stream_ref> stream_;
  cudax_hll inner_;
  std::uint8_t lg_config_k_;
  ::datasketches::target_hll_type tgt_;

  struct host_snapshot_t {
    std::vector<std::int32_t> registers;
    reduction_result reduction;
  };

  // ----- ctors -----
  //
  // Two overloads: own a stream on device 0, or borrow the caller's stream.

  // Owned stream on device 0.
  sketch_impl(std::uint8_t lgK, ::datasketches::target_hll_type tgt, MR mr)
    : stream_(std::in_place_type<::cuda::stream>, ::cuda::devices[0]),
      inner_(std::move(mr),
             precision{static_cast<int>(lgK)},
             policy_type{},
             std::get<::cuda::stream>(stream_)),
      lg_config_k_(lgK),
      tgt_(tgt)
  {
    check_target_(tgt);
  }

  // Borrowed stream. Caller's stream is recorded by cudax for later async-free.
  sketch_impl(std::uint8_t lgK,
              ::datasketches::target_hll_type tgt,
              MR mr,
              ::cuda::stream_ref user_stream)
    : stream_(user_stream),
      inner_(std::move(mr), precision{static_cast<int>(lgK)}, policy_type{}, user_stream),
      lg_config_k_(lgK),
      tgt_(tgt)
  {
    check_target_(tgt);
  }

  sketch_impl(const sketch_impl&)            = delete;
  sketch_impl& operator=(const sketch_impl&) = delete;
  sketch_impl(sketch_impl&&)                 = default;
  ~sketch_impl()                             = default;

  // Move assignment via swap-and-destroy. The defaulted member-wise version
  // would move-assign stream_ first; in the owned-stream case that destroys
  // *this*'s old CUDA stream before inner_ has a chance to release its
  // device buffer (which was allocated on that stream), producing
  // cudaFreeAsync against an invalid stream handle. The swap-based form
  // keeps each (stream_, inner_) pair intact: tmp takes over the incoming
  // sketch, swaps with *this*, then destructs the original *this* state
  // with its still-valid paired stream. Every swap is a handle/pointer
  // exchange -- no device copy or allocation. Works for both alternatives
  // of the variant (owned stream and borrowed stream_ref).
  sketch_impl& operator=(sketch_impl&& other) noexcept
  {
    if (this != &other) {
      sketch_impl tmp(std::move(other));
      using std::swap;
      swap(stream_, tmp.stream_);
      swap(inner_, tmp.inner_);
      swap(lg_config_k_, tmp.lg_config_k_);
      swap(tgt_, tmp.tgt_);
    }
    return *this;
  }

  // ----- stream accessor -----

  // Returns a stream_ref to whichever alternative the variant holds:
  // - cuda::stream  -> implicit conversion to stream_ref
  // - stream_ref    -> identity
  // The variant's invariants guarantee one alternative is always active.
  ::cuda::stream_ref stream() const noexcept
  {
    return std::visit([](const auto& s) noexcept -> ::cuda::stream_ref { return s; }, stream_);
  }

  // ----- internal helpers -----

  static void check_target_(::datasketches::target_hll_type tgt)
  {
    if (tgt != ::datasketches::HLL_8) {
      throw std::invalid_argument(
        "datasketches::cuda::hll_sketch supports only target_hll_type::HLL_8");
    }
  }

  // D2H copy of the register array + wider reduction. Used by everything that
  // reads the sketch state (estimate, bounds, is_empty, serialize). Returning
  // the host buffer lets `serialize_` avoid a second D2H.
  host_snapshot_t snapshot_(::cuda::stream_ref s) const
  {
    const std::size_t configK = std::size_t{1} << lg_config_k_;
    host_snapshot_t snap;
    snap.registers.resize(configK);
    const auto byte_span = inner_.sketch();
    DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(snap.registers.data(),
                                          byte_span.data(),
                                          configK * sizeof(std::int32_t),
                                          cudaMemcpyDeviceToHost,
                                          s.get()));
    DATASKETCHES_CUDA_TRY(cudaStreamSynchronize(s.get()));
    snap.reduction = reduce_hll8(
      ::cuda::std::span<const std::int32_t>{snap.registers.data(), snap.registers.size()},
      lg_config_k_);
    return snap;
  }

  static double estimate_from_(const reduction_result& r, std::uint8_t lgK)
  {
    return composite_finalizer(r.kxq0 + r.kxq1, r.cur_min, r.num_at_cur_min, lgK);
  }

  std::vector<std::uint8_t> serialize_(bool compact, ::cuda::stream_ref s) const
  {
    const std::size_t configK = std::size_t{1} << lg_config_k_;
    auto snap                 = snapshot_(s);

    preamble_fields f{};
    f.lgK                 = lg_config_k_;
    f.mode                = mode_hll;
    f.tgt                 = ::datasketches::HLL_8;
    f.is_empty            = (snap.reduction.num_at_cur_min == configK);
    f.is_compact          = compact;
    f.ooo_flag            = true;  // GPU forces OOO; pins CPU side to Composite.
    f.full_size_flag      = true;  // We always create the sketch in HLL mode.
    f.cur_min             = 0;     // HLL_8 invariant.
    f.num_at_cur_min      = snap.reduction.num_at_cur_min;
    f.kxq0                = snap.reduction.kxq0;
    f.kxq1                = snap.reduction.kxq1;
    f.hip_accum           = 0.0;  // GPU does not track HIP.
    f.aux_lg_int_arr_size = 0;
    f.aux_count           = 0;

    auto preamble = assemble_preamble(f);
    std::vector<std::uint8_t> out;
    out.reserve(PREAMBLE_BYTES + configK);
    out.insert(out.end(), preamble.begin(), preamble.end());
    for (std::size_t i = 0; i < configK; ++i) {
      out.push_back(static_cast<std::uint8_t>(snap.registers[i]));
    }
    return out;
  }

  // ----- deserialize support (called by the handle's static `deserialize`) -----

  // Parse + validate an HLL_8 blob preamble. Returns parsed fields; throws on
  // any unsupported mode/target or size mismatch.
  static preamble_fields parse_and_validate(::cuda::std::span<const std::uint8_t> bytes)
  {
    if (bytes.size() < PREAMBLE_BYTES) {
      throw std::invalid_argument(
        "datasketches::cuda::hll_sketch::deserialize: byte span shorter than 40-byte preamble");
    }
    ::cuda::std::span<const std::uint8_t, PREAMBLE_BYTES> head{bytes.data(),
                                                                   PREAMBLE_BYTES};
    // parse_preamble validates lgK against the supported range (4..21) before
    // returning, so the shift below is safe.
    const auto pf              = parse_preamble(head);
    const std::size_t configK  = std::size_t{1} << pf.lgK;
    const std::size_t expected = PREAMBLE_BYTES + configK;
    if (bytes.size() != expected) {
      throw std::invalid_argument(
        "datasketches::cuda::hll_sketch::deserialize: byte span size != preamble + 2^lgK");
    }
    return pf;
  }

  // H2D copy of the register bytes into this sketch's device buffer. Uses
  // the sketch's stream. Synchronizes so the registers are visible by return.
  void load_registers(::cuda::std::span<const std::uint8_t> bytes)
  {
    const std::size_t configK = std::size_t{1} << lg_config_k_;
    std::vector<std::int32_t> host_regs(configK);
    for (std::size_t i = 0; i < configK; ++i) {
      host_regs[i] = static_cast<std::int32_t>(bytes[PREAMBLE_BYTES + i]);
    }
    auto byte_span             = inner_.sketch();
    const ::cuda::stream_ref s = stream();
    DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(byte_span.data(),
                                          host_regs.data(),
                                          configK * sizeof(std::int32_t),
                                          cudaMemcpyHostToDevice,
                                          s.get()));
    DATASKETCHES_CUDA_TRY(cudaStreamSynchronize(s.get()));
  }

  // ----- operations -----

  template <class InputIt>
  void update(InputIt first, InputIt last)
  {
    inner_.add(first, last, stream());
  }

  template <class InputIt>
  void update(InputIt first, InputIt last, ::cuda::stream_ref s)
  {
    inner_.add(first, last, s);
  }

  template <class InputIt>
  void update_async(InputIt first, InputIt last)
  {
    inner_.add_async(first, last, stream());
  }

  template <class InputIt>
  void update_async(InputIt first, InputIt last, ::cuda::stream_ref s)
  {
    inner_.add_async(first, last, s);
  }

  double get_estimate() const { return get_estimate(stream()); }

  double get_estimate(::cuda::stream_ref s) const
  {
    return estimate_from_(snapshot_(s).reduction, lg_config_k_);
  }

  double get_lower_bound(std::uint8_t numStdDev) const
  {
    return get_lower_bound(numStdDev, stream());
  }

  double get_lower_bound(std::uint8_t numStdDev, ::cuda::stream_ref s) const
  {
    ::datasketches::HllUtil<>::checkNumStdDev(numStdDev);
    const auto rs               = snapshot_(s).reduction;
    const std::uint32_t configK = 1u << lg_config_k_;
    const double numNonZeros = (rs.cur_min == 0) ? static_cast<double>(configK - rs.num_at_cur_min)
                                                 : static_cast<double>(configK);
    const double estimate    = estimate_from_(rs, lg_config_k_);
    const double relErr      = ::datasketches::HllUtil<>::getRelErr(
      /*upperBound=*/false, /*unioned=*/true, lg_config_k_, numStdDev);
    return std::max(estimate / (1.0 + relErr), numNonZeros);
  }

  double get_upper_bound(std::uint8_t numStdDev) const
  {
    return get_upper_bound(numStdDev, stream());
  }

  double get_upper_bound(std::uint8_t numStdDev, ::cuda::stream_ref s) const
  {
    ::datasketches::HllUtil<>::checkNumStdDev(numStdDev);
    const double estimate = get_estimate(s);
    const double relErr   = ::datasketches::HllUtil<>::getRelErr(
      /*upperBound=*/true, /*unioned=*/true, lg_config_k_, numStdDev);
    return estimate / (1.0 + relErr);
  }

  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge(const sketch_impl<Key, OtherMR, OtherScope>& other)
  {
    inner_.merge(other.inner_, stream());
  }

  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge(const sketch_impl<Key, OtherMR, OtherScope>& other, ::cuda::stream_ref s)
  {
    inner_.merge(other.inner_, s);
  }

  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge_async(const sketch_impl<Key, OtherMR, OtherScope>& other)
  {
    inner_.merge_async(other.inner_, stream());
  }

  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge_async(const sketch_impl<Key, OtherMR, OtherScope>& other, ::cuda::stream_ref s)
  {
    inner_.merge_async(other.inner_, s);
  }

  std::uint8_t get_lg_config_k() const noexcept { return lg_config_k_; }
  ::datasketches::target_hll_type get_target_type() const noexcept { return tgt_; }

  bool is_empty() const { return is_empty(stream()); }

  bool is_empty(::cuda::stream_ref s) const
  {
    const auto rs = snapshot_(s).reduction;
    return rs.num_at_cur_min == (1u << lg_config_k_);
  }

  std::vector<std::uint8_t> serialize_compact() const
  {
    return serialize_(/*compact=*/true, stream());
  }

  std::vector<std::uint8_t> serialize_compact(::cuda::stream_ref s) const
  {
    return serialize_(/*compact=*/true, s);
  }

  std::vector<std::uint8_t> serialize_updatable() const
  {
    return serialize_(/*compact=*/false, stream());
  }

  std::vector<std::uint8_t> serialize_updatable(::cuda::stream_ref s) const
  {
    return serialize_(/*compact=*/false, s);
  }

  std::size_t num_registers() const noexcept { return std::size_t{1} << lg_config_k_; }
};

}  // namespace datasketches::cuda::detail::hll
