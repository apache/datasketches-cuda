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

#include <array>
#include <cstdint>
#include <cstring>
#include <cuda/std/span>
#include <stdexcept>

#include <HllUtil.hpp>
#include <hll.hpp>

namespace datasketches::cuda::detail::hll {

//! @brief Size of the HLL wire-format preamble in bytes.
constexpr std::size_t PREAMBLE_BYTES = 40;

//! Mode values stored in the low 2 bits of `MODE_BYTE` (byte 7).
//! Match `datasketches::hll_mode` values from
//! `HllSketchImpl-internal.hpp:makeModeByte`.
enum mode_byte : std::uint8_t {
  mode_list = 0,
  mode_set  = 1,
  mode_hll  = 2,
};

//! @brief High-level view of the 40-byte HLL preamble.
struct preamble_fields {
  std::uint8_t lgK;
  std::uint8_t mode;  // mode_byte
  std::uint8_t tgt;   // datasketches::cuda::target_hll_type
  bool is_empty;
  bool is_compact;
  bool ooo_flag;
  bool full_size_flag;
  std::uint8_t cur_min;
  std::uint32_t num_at_cur_min;
  double kxq0;
  double kxq1;
  double hip_accum;
  std::uint8_t aux_lg_int_arr_size;
  std::uint32_t aux_count;
};

//! @brief Serialize `f` into the 40-byte HLL preamble.
//!
//! Mirrors `HllArray::serialize` (`HllArray-internal.hpp:218-264`) byte-by-byte
//! for HLL mode. Constants come from `HllUtil.hpp` `hll_constants`.
//!
//! @param[in] f Preamble fields.
//! @return The 40-byte preamble.
inline std::array<std::uint8_t, PREAMBLE_BYTES> assemble_preamble(const preamble_fields& f)
{
  std::array<std::uint8_t, PREAMBLE_BYTES> b{};

  b[::datasketches::hll_constants::PREAMBLE_INTS_BYTE] = ::datasketches::hll_constants::HLL_PREINTS;
  b[::datasketches::hll_constants::SER_VER_BYTE]       = ::datasketches::hll_constants::SER_VER;
  b[::datasketches::hll_constants::FAMILY_BYTE]        = ::datasketches::hll_constants::FAMILY_ID;
  b[::datasketches::hll_constants::LG_K_BYTE]          = f.lgK;
  b[::datasketches::hll_constants::LG_ARR_BYTE]        = f.aux_lg_int_arr_size;

  std::uint8_t flags = 0;
  if (f.is_empty) flags |= ::datasketches::hll_constants::EMPTY_FLAG_MASK;
  if (f.is_compact) flags |= ::datasketches::hll_constants::COMPACT_FLAG_MASK;
  if (f.ooo_flag) flags |= ::datasketches::hll_constants::OUT_OF_ORDER_FLAG_MASK;
  if (f.full_size_flag) flags |= ::datasketches::hll_constants::FULL_SIZE_FLAG_MASK;
  b[::datasketches::hll_constants::FLAGS_BYTE] = flags;

  b[::datasketches::hll_constants::HLL_CUR_MIN_BYTE] = f.cur_min;
  b[::datasketches::hll_constants::MODE_BYTE] =
    static_cast<std::uint8_t>((f.tgt & 0x3u) << 2 | (f.mode & 0x3u));

  std::memcpy(&b[::datasketches::hll_constants::HIP_ACCUM_DOUBLE], &f.hip_accum, sizeof(double));
  std::memcpy(&b[::datasketches::hll_constants::KXQ0_DOUBLE], &f.kxq0, sizeof(double));
  std::memcpy(&b[::datasketches::hll_constants::KXQ1_DOUBLE], &f.kxq1, sizeof(double));
  std::memcpy(
    &b[::datasketches::hll_constants::CUR_MIN_COUNT_INT], &f.num_at_cur_min, sizeof(std::uint32_t));
  std::memcpy(
    &b[::datasketches::hll_constants::AUX_COUNT_INT], &f.aux_count, sizeof(std::uint32_t));

  return b;
}

//! @brief Parse a 40-byte HLL preamble.
//!
//! Only HLL mode + `HLL_8` target are supported. Throws on other modes/targets,
//! on bad SER_VER, or on bad FAMILY_ID.
//!
//! @param[in] bytes The 40 bytes to parse.
//! @return Parsed preamble fields.
inline preamble_fields parse_preamble(::cuda::std::span<const std::uint8_t, PREAMBLE_BYTES> bytes)
{
  if (bytes[::datasketches::hll_constants::SER_VER_BYTE] !=
      ::datasketches::hll_constants::SER_VER) {
    throw std::invalid_argument("datasketches::cuda preamble: bad SER_VER");
  }
  if (bytes[::datasketches::hll_constants::FAMILY_BYTE] !=
      ::datasketches::hll_constants::FAMILY_ID) {
    throw std::invalid_argument("datasketches::cuda preamble: bad FAMILY_ID");
  }
  if (bytes[::datasketches::hll_constants::PREAMBLE_INTS_BYTE] !=
      ::datasketches::hll_constants::HLL_PREINTS) {
    throw std::invalid_argument(
      "datasketches::cuda preamble: PREAMBLE_INTS != HLL_PREINTS (LIST/SET deserialize "
      "is not supported)");
  }

  preamble_fields f{};
  f.lgK = bytes[::datasketches::hll_constants::LG_K_BYTE];
  // LG_K_BYTE is an arbitrary input byte. Validate against the supported range
  // (datasketches-cpp HllUtil: 4..21) before any caller shifts on it -- otherwise
  // a malformed blob with lgK >= 64 would trigger UB at `1 << pf.lgK`. Throws
  // std::invalid_argument on out-of-range.
  ::datasketches::HllUtil<>::checkLgK(f.lgK);
  f.aux_lg_int_arr_size = bytes[::datasketches::hll_constants::LG_ARR_BYTE];
  const auto flags      = bytes[::datasketches::hll_constants::FLAGS_BYTE];
  f.is_empty            = (flags & ::datasketches::hll_constants::EMPTY_FLAG_MASK) != 0;
  f.is_compact          = (flags & ::datasketches::hll_constants::COMPACT_FLAG_MASK) != 0;
  f.ooo_flag            = (flags & ::datasketches::hll_constants::OUT_OF_ORDER_FLAG_MASK) != 0;
  f.full_size_flag      = (flags & ::datasketches::hll_constants::FULL_SIZE_FLAG_MASK) != 0;
  f.cur_min             = bytes[::datasketches::hll_constants::HLL_CUR_MIN_BYTE];

  const auto packed_mode_byte = bytes[::datasketches::hll_constants::MODE_BYTE];
  f.mode                      = static_cast<std::uint8_t>(packed_mode_byte & 0x3u);
  f.tgt                       = static_cast<std::uint8_t>((packed_mode_byte >> 2) & 0x3u);

  if (f.mode != mode_hll) {
    throw std::invalid_argument(
      "datasketches::cuda preamble: only HLL mode is supported (got LIST or SET)");
  }
  if (f.tgt != HLL_8) {
    throw std::invalid_argument("datasketches::cuda preamble: only HLL_8 target is supported");
  }

  std::memcpy(
    &f.hip_accum, &bytes[::datasketches::hll_constants::HIP_ACCUM_DOUBLE], sizeof(double));
  std::memcpy(&f.kxq0, &bytes[::datasketches::hll_constants::KXQ0_DOUBLE], sizeof(double));
  std::memcpy(&f.kxq1, &bytes[::datasketches::hll_constants::KXQ1_DOUBLE], sizeof(double));
  std::memcpy(&f.num_at_cur_min,
              &bytes[::datasketches::hll_constants::CUR_MIN_COUNT_INT],
              sizeof(std::uint32_t));
  std::memcpy(
    &f.aux_count, &bytes[::datasketches::hll_constants::AUX_COUNT_INT], sizeof(std::uint32_t));

  return f;
}

}  // namespace datasketches::cuda::detail::hll
