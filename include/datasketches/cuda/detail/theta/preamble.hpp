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
#include <cstring>
#include <cuda/std/span>
#include <stdexcept>
#include <vector>

#include <datasketches/cuda/detail/theta/policy.cuh>

#include <MurmurHash3.h>

namespace datasketches::cuda::detail::theta {

inline constexpr std::uint8_t uncompressed_serial_version = 3;
inline constexpr std::uint8_t sketch_type                 = 3;

inline constexpr std::uint8_t flag_big_endian = 1U << 0;
inline constexpr std::uint8_t flag_read_only  = 1U << 1;
inline constexpr std::uint8_t flag_empty      = 1U << 2;
inline constexpr std::uint8_t flag_compact    = 1U << 3;
inline constexpr std::uint8_t flag_ordered    = 1U << 4;

struct compact_image {
  bool empty;
  std::uint16_t seed_hash;
  std::uint64_t theta;
  std::vector<std::uint64_t> entries;
};

template <class T>
void write_at(std::vector<std::uint8_t>& bytes, std::size_t offset, const T& value)
{
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

template <class T>
T read_at(::cuda::std::span<const std::uint8_t> bytes, std::size_t offset)
{
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  return value;
}

inline std::vector<std::uint8_t> serialize_compact_v3(bool empty,
                                                      std::uint16_t seed_hash,
                                                      std::uint64_t theta,
                                                      const std::vector<std::uint64_t>& entries)
{
  const bool estimation             = !empty && theta < max_theta;
  const std::uint8_t preamble_longs = estimation ? 3 : (empty || entries.size() == 1 ? 1 : 2);
  const std::size_t preamble_bytes  = std::size_t{preamble_longs} * sizeof(std::uint64_t);

  std::vector<std::uint8_t> bytes(preamble_bytes + entries.size() * sizeof(std::uint64_t), 0);
  bytes[0] = preamble_longs;
  bytes[1] = uncompressed_serial_version;
  bytes[2] = sketch_type;
  bytes[5] = flag_read_only | flag_compact | flag_ordered | (empty ? flag_empty : 0);
  write_at(bytes, 6, seed_hash);
  if (preamble_longs > 1) { write_at(bytes, 8, static_cast<std::uint32_t>(entries.size())); }
  if (estimation) write_at(bytes, 16, theta);
  if (!entries.empty()) {
    std::memcpy(
      bytes.data() + preamble_bytes, entries.data(), entries.size() * sizeof(std::uint64_t));
  }
  return bytes;
}

inline compact_image parse_compact_v3(::cuda::std::span<const std::uint8_t> bytes,
                                      std::uint64_t seed)
{
  if (bytes.size() < sizeof(std::uint64_t)) {
    throw std::invalid_argument("theta_sketch::deserialize: image shorter than one preamble long");
  }

  const auto preamble_longs = bytes[0];
  if (preamble_longs < 1 || preamble_longs > 3) {
    throw std::invalid_argument("theta_sketch::deserialize: invalid preamble length");
  }
  const std::size_t preamble_bytes = std::size_t{preamble_longs} * sizeof(std::uint64_t);
  if (bytes.size() < preamble_bytes) {
    throw std::invalid_argument("theta_sketch::deserialize: truncated preamble");
  }

  if (bytes[1] != uncompressed_serial_version) {
    throw std::invalid_argument(
      "theta_sketch::deserialize supports only uncompressed serialization version 3");
  }
  if (bytes[2] != sketch_type) {
    throw std::invalid_argument("theta_sketch::deserialize: unexpected sketch type");
  }

  const auto flags = bytes[5];
  if ((flags & flag_big_endian) != 0) {
    throw std::invalid_argument("theta_sketch::deserialize: big-endian images are unsupported");
  }
  if ((flags & flag_compact) == 0 || (flags & flag_ordered) == 0) {
    throw std::invalid_argument(
      "theta_sketch::deserialize requires an ordered compact Theta image");
  }

  const bool empty              = (flags & flag_empty) != 0;
  const std::uint16_t seed_hash = read_at<std::uint16_t>(bytes, 6);
  if (!empty && seed_hash != ::compute_seed_hash(seed)) {
    throw std::invalid_argument("theta_sketch::deserialize: seed hash mismatch");
  }

  std::uint32_t count = 0;
  std::uint64_t theta = max_theta;
  if (empty) {
    if (preamble_longs != 1) {
      throw std::invalid_argument("theta_sketch::deserialize: empty image has invalid preamble");
    }
  } else if (preamble_longs == 1) {
    count = 1;
  } else {
    count = read_at<std::uint32_t>(bytes, 8);
    if (preamble_longs == 3) theta = read_at<std::uint64_t>(bytes, 16);
  }

  if (theta == 0 || theta > max_theta) {
    throw std::invalid_argument("theta_sketch::deserialize: theta is out of range");
  }
  if (preamble_longs == 3 && theta == max_theta) {
    throw std::invalid_argument("theta_sketch::deserialize: estimation image has maximum theta");
  }

  const std::size_t expected = preamble_bytes + std::size_t{count} * sizeof(std::uint64_t);
  if (bytes.size() != expected) {
    throw std::invalid_argument("theta_sketch::deserialize: image size mismatch");
  }

  compact_image image{empty, seed_hash, theta, std::vector<std::uint64_t>(count)};
  if (count != 0) {
    std::memcpy(image.entries.data(), bytes.data() + preamble_bytes, count * sizeof(std::uint64_t));
  }
  for (std::size_t i = 0; i < image.entries.size(); ++i) {
    const auto hash = image.entries[i];
    if (hash == 0 || hash >= theta) {
      throw std::invalid_argument("theta_sketch::deserialize: retained hash is out of range");
    }
    if (i != 0 && image.entries[i - 1] >= hash) {
      throw std::invalid_argument(
        "theta_sketch::deserialize: retained hashes are not strictly ordered");
    }
  }
  return image;
}

}  // namespace datasketches::cuda::detail::theta
