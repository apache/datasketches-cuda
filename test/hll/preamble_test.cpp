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

// Validates the 40-byte HLL preamble assembler/parser via:
//   1. Round-trip: assemble(parse(b)) == b for valid inputs.
//   2. Byte-compare against `datasketches::hll_sketch::serialize_compact()` of
//      a CPU HLL_8 sketch in HLL mode (start_full_size=true).

#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <hll.hpp>

#include <datasketches/cuda/detail/hll/preamble.hpp>

using datasketches::cuda::detail::hll::PREAMBLE_BYTES;
using datasketches::cuda::detail::hll::preamble_fields;

TEST_CASE("preamble round-trip", "[preamble]")
{
  preamble_fields f{};
  f.lgK                 = 12;
  f.mode                = datasketches::cuda::detail::hll::mode_hll;
  f.tgt                 = ::datasketches::HLL_8;
  f.is_empty            = false;
  f.is_compact          = true;
  f.ooo_flag            = true;
  f.full_size_flag      = true;
  f.cur_min             = 0;
  f.num_at_cur_min      = 1234;
  f.kxq0                = 4096.5;
  f.kxq1                = 0.125;
  f.hip_accum           = 0.0;
  f.aux_lg_int_arr_size = 0;
  f.aux_count           = 0;

  auto bytes = datasketches::cuda::detail::hll::assemble_preamble(f);
  ::cuda::std::span<const std::uint8_t, PREAMBLE_BYTES> sp{bytes.data(), PREAMBLE_BYTES};
  auto parsed = datasketches::cuda::detail::hll::parse_preamble(sp);

  REQUIRE(parsed.lgK == f.lgK);
  REQUIRE(parsed.mode == f.mode);
  REQUIRE(parsed.tgt == f.tgt);
  REQUIRE(parsed.is_empty == f.is_empty);
  REQUIRE(parsed.is_compact == f.is_compact);
  REQUIRE(parsed.ooo_flag == f.ooo_flag);
  REQUIRE(parsed.full_size_flag == f.full_size_flag);
  REQUIRE(parsed.cur_min == f.cur_min);
  REQUIRE(parsed.num_at_cur_min == f.num_at_cur_min);
  REQUIRE(parsed.kxq0 == f.kxq0);
  REQUIRE(parsed.kxq1 == f.kxq1);
  REQUIRE(parsed.hip_accum == f.hip_accum);
  REQUIRE(parsed.aux_count == f.aux_count);
}

TEST_CASE("preamble byte-compatible with datasketches::hll_sketch", "[preamble]")
{
  for (uint8_t lgK : {uint8_t{8}, uint8_t{12}, uint8_t{16}}) {
    ::datasketches::hll_sketch cpu(lgK, ::datasketches::HLL_8, /*start_full_size=*/true);
    std::mt19937_64 rng(0xCAFEBABEULL ^ lgK);
    for (uint64_t i = 0; i < 50'000; ++i)
      cpu.update(rng());
    auto cpu_bytes = cpu.serialize_compact();

    // Parse the CPU bytes via our parser.
    ::cuda::std::span<const std::uint8_t, PREAMBLE_BYTES> head{cpu_bytes.data(), PREAMBLE_BYTES};
    auto parsed = datasketches::cuda::detail::hll::parse_preamble(head);

    // Re-assemble from the parsed fields.
    auto round = datasketches::cuda::detail::hll::assemble_preamble(parsed);

    INFO("lgK=" << int(lgK));
    for (std::size_t i = 0; i < PREAMBLE_BYTES; ++i) {
      INFO("byte " << i);
      REQUIRE(round[i] == cpu_bytes[i]);
    }
  }
}

TEST_CASE("preamble parser rejects unsupported modes/targets", "[preamble]")
{
  // Build a valid HLL_8 preamble, then mutate the MODE byte to LIST/SET or
  // tgt to HLL_4/HLL_6 and confirm the parser throws.
  preamble_fields f{};
  f.lgK      = 12;
  f.mode     = datasketches::cuda::detail::hll::mode_hll;
  f.tgt      = ::datasketches::HLL_8;
  auto bytes = datasketches::cuda::detail::hll::assemble_preamble(f);

  // mode = LIST
  bytes[::datasketches::hll_constants::MODE_BYTE] =
    static_cast<uint8_t>((::datasketches::HLL_8 << 2) | datasketches::cuda::detail::hll::mode_list);
  // PREAMBLE_INTS == HLL_PREINTS still, so we hit the mode-check branch.
  // (Note: a real LIST blob would carry PREAMBLE_INTS=2, but we check the mode
  // byte path here which is the deserialize gate for non-HLL modes.)
  // The PREAMBLE_INTS check would also fire for a LIST blob. To exercise the
  // mode-check specifically, we keep PREAMBLE_INTS=HLL_PREINTS and rely on the
  // mode byte being non-HLL.
  REQUIRE_THROWS_AS(
    datasketches::cuda::detail::hll::parse_preamble(
      ::cuda::std::span<const std::uint8_t, PREAMBLE_BYTES>{bytes.data(), PREAMBLE_BYTES}),
    std::invalid_argument);

  // tgt = HLL_4
  auto bytes_hll4 = datasketches::cuda::detail::hll::assemble_preamble(f);
  bytes_hll4[::datasketches::hll_constants::MODE_BYTE] =
    static_cast<uint8_t>((::datasketches::HLL_4 << 2) | datasketches::cuda::detail::hll::mode_hll);
  REQUIRE_THROWS_AS(
    datasketches::cuda::detail::hll::parse_preamble(
      ::cuda::std::span<const std::uint8_t, PREAMBLE_BYTES>{bytes_hll4.data(), PREAMBLE_BYTES}),
    std::invalid_argument);
}

TEST_CASE("preamble parser rejects out-of-range lgK", "[preamble]")
{
  // Build a valid HLL_8 preamble, then overwrite the lgK byte with values
  // outside HllUtil's [4, 21] range. parse_preamble must throw before any
  // caller shifts `1 << lgK` (avoiding UB for lgK >= 64).
  preamble_fields f{};
  f.lgK      = 12;
  f.mode     = datasketches::cuda::detail::hll::mode_hll;
  f.tgt      = ::datasketches::HLL_8;
  auto bytes = datasketches::cuda::detail::hll::assemble_preamble(f);

  for (uint8_t bad_lgK : {uint8_t{0}, uint8_t{3}, uint8_t{22}, uint8_t{64}, uint8_t{255}}) {
    INFO("bad lgK = " << int(bad_lgK));
    bytes[::datasketches::hll_constants::LG_K_BYTE] = bad_lgK;
    REQUIRE_THROWS_AS(
      datasketches::cuda::detail::hll::parse_preamble(
        ::cuda::std::span<const std::uint8_t, PREAMBLE_BYTES>{bytes.data(), PREAMBLE_BYTES}),
      std::invalid_argument);
  }
}
