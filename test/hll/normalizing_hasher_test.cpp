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

// Host-only unit tests for normalizing_hasher<Key>::_canonicalize.
//
// These run without a GPU and verify that each normalization branch produces
// the canonical uint64_t bit pattern matching datasketches-cpp's host-side
// update(...) overloads. The tests are intentionally white-box: they compare
// hash outputs between types that datasketches-cpp maps to the same canonical
// representation (e.g. int32_t(-1) == uint32_t(0xFFFFFFFF) after sign-extension).

#include <cstdint>
#include <cstring>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/detail/hll/normalizing_hasher.cuh>

using datasketches::cuda::detail::hll::normalizing_hasher;

// Helper: invoke normalizing_hasher with seed 9001 (same as all tests).
template <class T>
static __uint128_t h(T v)
{
  return normalizing_hasher<T>{}(v);
}

// ============================================================
// Integral sign-extension parity
// ============================================================

TEST_CASE("int32/16/8 sign-extend to same int64 representation", "[normalizing_hasher]")
{
  // int32_t(-1) sign-extends to int64_t(-1); all three signed narrow types do.
  REQUIRE(h<int32_t>(-1) == h<int64_t>(-1));
  REQUIRE(h<int16_t>(-1) == h<int64_t>(-1));
  REQUIRE(h<int8_t>(-1) == h<int64_t>(-1));

  REQUIRE(h<int32_t>(1) == h<int64_t>(1));
  REQUIRE(h<int16_t>(1) == h<int64_t>(1));
  REQUIRE(h<int8_t>(1) == h<int64_t>(1));

  // INT32_MIN sign-extends to INT64_MIN.
  REQUIRE(h<int32_t>(std::numeric_limits<int32_t>::min()) ==
          h<int64_t>(static_cast<int64_t>(std::numeric_limits<int32_t>::min())));
}

TEST_CASE("uint narrow types reinterpret-as-signed then sign-extend", "[normalizing_hasher]")
{
  // update(uint32_t) → update(int32_t) → sign-extend to int64_t
  // So uint32_t(0xFFFFFFFF) == uint32_t bit-pattern of int32_t(-1) → int64_t(-1)
  REQUIRE(h<uint32_t>(0xFFFFFFFFu) == h<int64_t>(-1));
  REQUIRE(h<uint16_t>(0xFFFFu) == h<int64_t>(-1));
  REQUIRE(h<uint8_t>(0xFFu) == h<int64_t>(-1));

  // uint32_t(0) == int64_t(0)
  REQUIRE(h<uint32_t>(0u) == h<int64_t>(0));
  REQUIRE(h<uint16_t>(0u) == h<int64_t>(0));
  REQUIRE(h<uint8_t>(0u) == h<int64_t>(0));

  // uint32_t(1) == int64_t(1) (no reinterpretation needed, same bits)
  REQUIRE(h<uint32_t>(1u) == h<int64_t>(1));

  // uint32_t(0x80000000) reinterprets as int32_t(-2147483648) then sign-extends
  REQUIRE(h<uint32_t>(0x80000000u) ==
          h<int64_t>(static_cast<int64_t>(static_cast<int32_t>(0x80000000u))));
}

TEST_CASE("uint64_t and int64_t with same bit pattern produce same hash", "[normalizing_hasher]")
{
  // Both hash 8 raw bytes -- no normalization for 64-bit types.
  REQUIRE(h<uint64_t>(0) == h<int64_t>(0));
  REQUIRE(h<uint64_t>(1) == h<int64_t>(1));
  // 0xFFFFFFFFFFFFFFFF == int64_t(-1) as bit pattern
  REQUIRE(h<uint64_t>(UINT64_MAX) == h<int64_t>(-1));
}

// ============================================================
// double / float canonicalization
// ============================================================

TEST_CASE("double: -0.0 and +0.0 produce the same hash", "[normalizing_hasher]")
{
  const double pos_zero = 0.0;
  double neg_zero       = 0.0;
  neg_zero              = -neg_zero;
  REQUIRE(h<double>(pos_zero) == h<double>(neg_zero));
}

TEST_CASE("double: all NaN payloads produce the same hash", "[normalizing_hasher]")
{
  // Construct distinct NaN bit patterns (positive quiet, negative quiet,
  // signaling-like with different payloads).
  auto bits_to_double = [](uint64_t b) {
    double d;
    std::memcpy(&d, &b, 8);
    return d;
  };
  const double nan_a = bits_to_double(0x7ff8000000000001ULL);  // positive quiet NaN
  const double nan_b = bits_to_double(0xfff8000000000001ULL);  // negative quiet NaN
  const double nan_c = bits_to_double(0x7ff0000000000001ULL);  // signaling-ish NaN
  REQUIRE(h<double>(nan_a) == h<double>(nan_b));
  REQUIRE(h<double>(nan_b) == h<double>(nan_c));
}

TEST_CASE("double: NaN hash equals the Java canonical 0x7ff8000000000000", "[normalizing_hasher]")
{
  auto bits_to_double = [](uint64_t b) {
    double d;
    std::memcpy(&d, &b, 8);
    return d;
  };
  const double canonical_nan = bits_to_double(0x7ff8000000000000ULL);
  const double other_nan     = bits_to_double(0x7ff8000000000042ULL);
  // Both NaNs must hash the same as the canonical form itself.
  REQUIRE(h<double>(other_nan) == h<double>(canonical_nan));
}

TEST_CASE("float: -0.0f and +0.0f produce the same hash", "[normalizing_hasher]")
{
  const float pos_zero = 0.0f;
  float neg_zero       = 0.0f;
  neg_zero             = -neg_zero;
  REQUIRE(h<float>(pos_zero) == h<float>(neg_zero));
}

TEST_CASE("float: all NaN payloads produce the same hash", "[normalizing_hasher]")
{
  auto bits_to_float = [](uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
  };
  const float nan_a = bits_to_float(0x7fc00001u);  // positive quiet NaN
  const float nan_b = bits_to_float(0xffc00001u);  // negative quiet NaN
  REQUIRE(h<float>(nan_a) == h<float>(nan_b));
}

TEST_CASE("float NaN hashes same as double NaN (both canonicalize to 0x7ff8000000000000)",
          "[normalizing_hasher]")
{
  auto bits_to_float = [](uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
  };
  auto bits_to_double = [](uint64_t b) {
    double d;
    std::memcpy(&d, &b, 8);
    return d;
  };
  const float fnan  = bits_to_float(0x7fc00001u);
  const double dnan = bits_to_double(0x7ff8000000000000ULL);  // the canonical form
  // float NaN → double NaN → canonical 0x7ff8000000000000 → same hash as dnan
  REQUIRE(h<float>(fnan) == h<double>(dnan));
}

TEST_CASE("float -0.0f hashes same as double -0.0", "[normalizing_hasher]")
{
  float neg_zero_f  = -0.0f;
  double neg_zero_d = -0.0;
  // Both canonicalize to +0.0 before hashing.
  REQUIRE(h<float>(neg_zero_f) == h<double>(neg_zero_d));
}
