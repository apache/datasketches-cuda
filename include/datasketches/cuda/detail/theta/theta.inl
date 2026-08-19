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

namespace datasketches::cuda {

template <class Key, class MR>
theta_sketch<Key, MR>::theta_sketch(
  ::cuda::stream_ref stream, MR mr, std::uint8_t lg_k, std::uint64_t seed, float p)
  : impl_(stream, std::move(mr), lg_k, seed, p)
{
}

template <class Key, class MR>
template <class RandomAccessIt>
void theta_sketch<Key, MR>::update(::cuda::stream_ref stream,
                                   RandomAccessIt first,
                                   RandomAccessIt last)
{
  impl_.update(stream, first, last);
}

template <class Key, class MR>
template <class OtherMR>
void theta_sketch<Key, MR>::merge(::cuda::stream_ref stream,
                                  const theta_sketch<Key, OtherMR>& other)
{
  impl_.merge(stream, other.impl_);
}

template <class Key, class MR>
template <class OtherMR>
void theta_sketch<Key, MR>::intersect(::cuda::stream_ref stream,
                                      const theta_sketch<Key, OtherMR>& other)
{
  impl_.intersect(stream, other.impl_);
}

template <class Key, class MR>
template <class OtherMR>
void theta_sketch<Key, MR>::a_not_b(::cuda::stream_ref stream,
                                    const theta_sketch<Key, OtherMR>& other)
{
  impl_.a_not_b(stream, other.impl_);
}

template <class Key, class MR>
void theta_sketch<Key, MR>::reset(::cuda::stream_ref stream)
{
  impl_.reset(stream);
}

template <class Key, class MR>
bool theta_sketch<Key, MR>::is_empty() const noexcept
{
  return impl_.is_empty();
}

template <class Key, class MR>
bool theta_sketch<Key, MR>::is_estimation_mode() const noexcept
{
  return impl_.is_estimation_mode();
}

template <class Key, class MR>
bool theta_sketch<Key, MR>::is_ordered() const noexcept
{
  return true;
}

template <class Key, class MR>
std::uint8_t theta_sketch<Key, MR>::get_lg_k() const noexcept
{
  return impl_.get_lg_k();
}

template <class Key, class MR>
std::uint64_t theta_sketch<Key, MR>::get_theta64() const noexcept
{
  return impl_.get_theta64();
}

template <class Key, class MR>
double theta_sketch<Key, MR>::get_theta() const noexcept
{
  return impl_.get_theta();
}

template <class Key, class MR>
std::uint16_t theta_sketch<Key, MR>::get_seed_hash() const noexcept
{
  return impl_.get_seed_hash();
}

template <class Key, class MR>
std::size_t theta_sketch<Key, MR>::get_num_retained() const noexcept
{
  return impl_.get_num_retained();
}

template <class Key, class MR>
double theta_sketch<Key, MR>::get_estimate() const noexcept
{
  return impl_.get_estimate();
}

template <class Key, class MR>
double theta_sketch<Key, MR>::get_lower_bound(std::uint8_t num_std_devs) const
{
  return impl_.get_lower_bound(num_std_devs);
}

template <class Key, class MR>
double theta_sketch<Key, MR>::get_upper_bound(std::uint8_t num_std_devs) const
{
  return impl_.get_upper_bound(num_std_devs);
}

template <class Key, class MR>
std::vector<typename theta_sketch<Key, MR>::hash_type> theta_sketch<Key, MR>::get_retained_hashes(
  ::cuda::stream_ref stream) const
{
  return impl_.get_retained_hashes(stream);
}

template <class Key, class MR>
std::vector<std::uint8_t> theta_sketch<Key, MR>::serialize_compact(::cuda::stream_ref stream) const
{
  return impl_.serialize_compact(stream);
}

template <class Key, class MR>
theta_sketch<Key, MR> theta_sketch<Key, MR>::deserialize(
  ::cuda::stream_ref stream,
  ::cuda::std::span<const std::uint8_t> bytes,
  MR mr,
  std::uint8_t lg_k,
  std::uint64_t seed)
{
  auto image = detail::theta::parse_compact_v3(bytes, seed);
  theta_sketch sketch(stream, std::move(mr), lg_k, seed);
  sketch.impl_.load_compact_(stream, image);
  return sketch;
}

}  // namespace datasketches::cuda
