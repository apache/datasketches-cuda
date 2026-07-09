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

// CPU(N keys, start_full_size=true) -> bytes -> GPU.deserialize -> GPU.serialize -> bytes.
// Compare per-byte region:
//   - bytes 0-4: exact (preamble header + lgK + lg_arr=0)
//   - byte 5 FLAGS: ignored (CPU has oooFlag=false from incremental update,
//                    GPU forces oooFlag=true on serialize)
//   - byte 6 cur_min: exact (both write 0 for HLL_8)
//   - byte 7 mode: exact
//   - bytes 8-15 hip_accum: ignored (CPU has live value, GPU writes 0)
//   - bytes 16-39: exact (kxq0, kxq1, num_at_cur_min, aux_count) -- deterministic
//                  functions of the register array
//   - bytes 40+: exact (register array)

#include <cstdint>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <hll.hpp>

#include <datasketches/cuda/hll.hpp>

namespace {

// Deterministic CPU sketch in HLL mode for `n` random keys.
std::vector<std::uint8_t> cpu_bytes(uint8_t lgK, uint64_t n, uint64_t seed)
{
  ::datasketches::hll_sketch cpu(lgK, ::datasketches::HLL_8, /*start_full_size=*/true);
  std::mt19937_64 rng(seed);
  for (uint64_t i = 0; i < n; ++i)
    cpu.update(rng());
  return cpu.serialize_compact();
}

void compare_round_trip(const std::vector<uint8_t>& cpu, const std::vector<uint8_t>& gpu)
{
  REQUIRE(cpu.size() == gpu.size());

  // bytes 0-4: exact
  for (std::size_t i = 0; i <= 4; ++i) {
    INFO("byte " << i);
    REQUIRE(cpu[i] == gpu[i]);
  }

  // byte 5 FLAGS: skip (GPU forces ooo=1)
  // bytes 6-7 cur_min, mode: exact
  REQUIRE(cpu[6] == gpu[6]);
  REQUIRE(cpu[7] == gpu[7]);

  // bytes 8-15 hip_accum: skip (GPU writes 0)

  // bytes 16-39: exact
  for (std::size_t i = 16; i < 40; ++i) {
    INFO("byte " << i);
    REQUIRE(cpu[i] == gpu[i]);
  }

  // bytes 40+: register array exact
  for (std::size_t i = 40; i < cpu.size(); ++i) {
    if (cpu[i] != gpu[i]) {
      INFO("register byte " << (i - 40));
      REQUIRE(cpu[i] == gpu[i]);
    }
  }
}

}  // namespace

TEST_CASE("CPU -> GPU.deserialize -> GPU.serialize round-trip", "[round_trip]")
{
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  for (uint8_t lgK : {uint8_t{8}, uint8_t{12}, uint8_t{16}}) {
    const uint64_t n = (uint64_t{1} << lgK) * 64;
    auto cpu         = cpu_bytes(lgK, n, 0xC0DEBA5EULL ^ lgK);
    auto gpu         = datasketches::cuda::hll_sketch<uint64_t>::deserialize(
      stream, ::cuda::std::span<const std::uint8_t>{cpu.data(), cpu.size()}, mr);
    auto round = gpu.serialize_compact(stream);
    INFO("lgK=" << int(lgK) << " n=" << n);
    compare_round_trip(cpu, round);
  }
}

TEST_CASE("GPU.serialize FLAGS forces oooFlag=true", "[round_trip]")
{
  const uint8_t lgK = 12;
  auto cpu          = cpu_bytes(lgK, 1u << (lgK + 4), 0xBADB100D42ULL);
  // CPU bytes are from incremental update with start_full_size, so oooFlag=false.
  REQUIRE((cpu[5] & 0x10) == 0);

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr  = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  auto gpu = datasketches::cuda::hll_sketch<uint64_t>::deserialize(
    stream, ::cuda::std::span<const std::uint8_t>{cpu.data(), cpu.size()}, mr);
  auto round = gpu.serialize_compact(stream);
  // GPU re-serialize forces oooFlag=true.
  REQUIRE((round[5] & 0x10) != 0);

  // GPU re-serialize forces hip_accum=0 regardless of input.
  double hip_round{};
  std::memcpy(&hip_round, round.data() + 8, sizeof(double));
  REQUIRE(hip_round == 0.0);
}
