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
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <cuda/experimental/__cuco/hyperloglog.cuh>

#include <hll.hpp>

#include <datasketches/cuda/detail/common/error.hpp>
#include <datasketches/cuda/detail/hll/policy.cuh>
#include <datasketches/cuda/detail/hll/preamble.hpp>
#include <datasketches/cuda/detail/hll/reduction_state.hpp>
#include <datasketches/cuda/detail/hll/sketch_ref_impl.cuh>

namespace datasketches::cuda::detail::hll {

// Owning implementation behind `datasketches::cuda::hll_sketch`. It owns the
// cudax HyperLogLog, precision/target metadata, and serialization machinery.
// Shared sketch operations delegate through an on-demand `sketch_ref_impl`.
//
// Declared as a `struct` because everything is intended for use by the public
// handle or by other implementation methods.
//
// Stream lifetime: the caller must keep the stream supplied at construction
// alive until the sketch is destroyed. The backing cudax object may use that
// stream for async deallocation. Any streams used with async operations must
// also be synchronized or ordered before destruction.
template <class Key,
          class MR                   = ::cuda::device_memory_pool_ref,
          ::cuda::thread_scope Scope = ::cuda::thread_scope_device>
struct sketch_impl {
  using key_type      = Key;
  using policy_type   = policy<Key>;
  using register_type = typename policy_type::register_type;
  using ref_impl_type = sketch_ref_impl<Key, Scope>;

  using cudax_hll = ::cuda::experimental::cuco::hyperloglog<Key, MR, Scope, policy_type>;
  using precision = typename cudax_hll::precision;

  std::uint8_t lg_config_k_;
  ::datasketches::target_hll_type tgt_;
  cudax_hll inner_;

  struct host_snapshot_t {
    std::vector<register_type> registers;
    reduction_result reduction;
  };

  sketch_impl(::cuda::stream_ref stream,
              MR mr,
              std::uint8_t lgK,
              ::datasketches::target_hll_type tgt)
    : lg_config_k_(lgK),
      tgt_(check_target_(tgt)),
      inner_(stream, std::move(mr), precision{static_cast<int>(lgK)}, policy_type{})
  {
  }

  sketch_impl(const sketch_impl&)            = delete;
  sketch_impl& operator=(const sketch_impl&) = delete;
  sketch_impl(sketch_impl&&)                 = default;
  ~sketch_impl()                             = default;
  sketch_impl& operator=(sketch_impl&&)      = default;

  // ----- internal helpers -----

  static ::datasketches::target_hll_type check_target_(::datasketches::target_hll_type tgt)
  {
    if (tgt != ::datasketches::HLL_8) {
      throw std::invalid_argument(
        "datasketches::cuda::hll_sketch supports only target_hll_type::HLL_8");
    }
    return tgt;
  }

  // D2H copy of the register array + wider reduction used by serialization.
  host_snapshot_t snapshot_(::cuda::stream_ref s) const
  {
    const std::size_t configK = std::size_t{1} << lg_config_k_;
    host_snapshot_t snap;
    snap.registers.resize(configK);
    const auto byte_span = ref().sketch();
    DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(snap.registers.data(),
                                          byte_span.data(),
                                          configK * sizeof(register_type),
                                          cudaMemcpyDeviceToHost,
                                          s.get()));
    s.sync();
    snap.reduction = reduce_hll8(
      ::cuda::std::span<const register_type>{snap.registers.data(), snap.registers.size()},
      lg_config_k_);
    return snap;
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
    ::cuda::std::span<const std::uint8_t, PREAMBLE_BYTES> head{bytes.data(), PREAMBLE_BYTES};
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

  // H2D copy of the register bytes into this sketch's device buffer.
  // Synchronizes so the registers are visible by return.
  void load_registers(::cuda::stream_ref s, ::cuda::std::span<const std::uint8_t> bytes)
  {
    const std::size_t configK = std::size_t{1} << lg_config_k_;
    std::vector<register_type> host_regs(configK);
    for (std::size_t i = 0; i < configK; ++i) {
      const auto value = bytes[PREAMBLE_BYTES + i];
      if (value > 63) {
        throw std::invalid_argument(
          "datasketches::cuda::hll_sketch::deserialize: HLL_8 register value out of range");
      }
      host_regs[i] = static_cast<register_type>(value);
    }
    auto byte_span = ref().sketch();
    DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(byte_span.data(),
                                          host_regs.data(),
                                          configK * sizeof(register_type),
                                          cudaMemcpyHostToDevice,
                                          s.get()));
    s.sync();
  }

  // ----- operations -----

  template <class InputIt>
  void update(::cuda::stream_ref s, InputIt first, InputIt last)
  {
    ref().update(s, first, last);
  }

  template <class InputIt>
  void update_async(::cuda::stream_ref s, InputIt first, InputIt last)
  {
    ref().update_async(s, first, last);
  }

  [[nodiscard]] ref_impl_type ref() const noexcept { return ref_impl_type{inner_.ref()}; }

  double get_estimate(::cuda::stream_ref s) const
  {
    return static_cast<double>(ref().get_estimate(s));
  }

  double get_lower_bound(::cuda::stream_ref s, std::uint8_t numStdDev) const
  {
    return ref().get_lower_bound(s, numStdDev);
  }

  double get_upper_bound(::cuda::stream_ref s, std::uint8_t numStdDev) const
  {
    return ref().get_upper_bound(s, numStdDev);
  }

  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge(::cuda::stream_ref s, const sketch_impl<Key, OtherMR, OtherScope>& other)
  {
    ref().merge(s, other.ref());
  }

  template <class OtherMR, ::cuda::thread_scope OtherScope>
  void merge_async(::cuda::stream_ref s, const sketch_impl<Key, OtherMR, OtherScope>& other)
  {
    ref().merge_async(s, other.ref());
  }

  std::uint8_t get_lg_config_k() const noexcept { return ref().get_lg_config_k(); }
  ::datasketches::target_hll_type get_target_type() const noexcept
  {
    return ref().get_target_type();
  }

  bool is_empty(::cuda::stream_ref s) const { return ref().is_empty(s); }

  std::vector<std::uint8_t> serialize_compact(::cuda::stream_ref s) const
  {
    return serialize_(/*compact=*/true, s);
  }

  std::vector<std::uint8_t> serialize_updatable(::cuda::stream_ref s) const
  {
    return serialize_(/*compact=*/false, s);
  }

  std::size_t num_registers() const noexcept { return ref().num_registers(); }
};

}  // namespace datasketches::cuda::detail::hll
