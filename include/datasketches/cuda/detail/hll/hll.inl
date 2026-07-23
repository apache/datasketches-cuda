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

namespace datasketches::cuda {

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

template <class Key, class MR, ::cuda::thread_scope Scope>
hll_sketch<Key, MR, Scope>::hll_sketch(::cuda::stream_ref stream,
                                       MR mr,
                                       std::uint8_t lgK,
                                       target_hll_type tgt)
  : impl_(stream, std::move(mr), lgK, tgt)
{
}

// ---------------------------------------------------------------------------
// Stream / update
// ---------------------------------------------------------------------------

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class InputIt>
void hll_sketch<Key, MR, Scope>::update(::cuda::stream_ref stream, InputIt first, InputIt last)
{
  impl_.update(stream, first, last);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class InputIt>
void hll_sketch<Key, MR, Scope>::update_async(::cuda::stream_ref stream,
                                              InputIt first,
                                              InputIt last)
{
  impl_.update_async(stream, first, last);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
typename hll_sketch<Key, MR, Scope>::ref_type hll_sketch<Key, MR, Scope>::ref() noexcept
{
  return ref_type{impl_.ref()};
}

// ---------------------------------------------------------------------------
// Estimate and bounds
// ---------------------------------------------------------------------------

template <class Key, class MR, ::cuda::thread_scope Scope>
double hll_sketch<Key, MR, Scope>::get_estimate(::cuda::stream_ref stream) const
{
  return impl_.get_estimate(stream);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
double hll_sketch<Key, MR, Scope>::get_lower_bound(::cuda::stream_ref stream,
                                                   std::uint8_t numStdDev) const
{
  return impl_.get_lower_bound(stream, numStdDev);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
double hll_sketch<Key, MR, Scope>::get_upper_bound(::cuda::stream_ref stream,
                                                   std::uint8_t numStdDev) const
{
  return impl_.get_upper_bound(stream, numStdDev);
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class OtherMR, ::cuda::thread_scope OtherScope>
void hll_sketch<Key, MR, Scope>::merge(::cuda::stream_ref stream,
                                       const hll_sketch<Key, OtherMR, OtherScope>& other)
{
  impl_.merge(stream, other.impl_);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class OtherMR, ::cuda::thread_scope OtherScope>
void hll_sketch<Key, MR, Scope>::merge_async(::cuda::stream_ref stream,
                                             const hll_sketch<Key, OtherMR, OtherScope>& other)
{
  impl_.merge_async(stream, other.impl_);
}

// ---------------------------------------------------------------------------
// State accessors
// ---------------------------------------------------------------------------

template <class Key, class MR, ::cuda::thread_scope Scope>
std::uint8_t hll_sketch<Key, MR, Scope>::get_lg_config_k() const noexcept
{
  return impl_.get_lg_config_k();
}

template <class Key, class MR, ::cuda::thread_scope Scope>
target_hll_type hll_sketch<Key, MR, Scope>::get_target_type() const noexcept
{
  return impl_.get_target_type();
}

template <class Key, class MR, ::cuda::thread_scope Scope>
bool hll_sketch<Key, MR, Scope>::is_empty(::cuda::stream_ref stream) const
{
  return impl_.is_empty(stream);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
std::size_t hll_sketch<Key, MR, Scope>::num_registers() const noexcept
{
  return impl_.num_registers();
}

// ---------------------------------------------------------------------------
// Serialize / deserialize
// ---------------------------------------------------------------------------

template <class Key, class MR, ::cuda::thread_scope Scope>
std::vector<std::uint8_t> hll_sketch<Key, MR, Scope>::serialize_compact(
  ::cuda::stream_ref stream) const
{
  return impl_.serialize_compact(stream);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
std::vector<std::uint8_t> hll_sketch<Key, MR, Scope>::serialize_updatable(
  ::cuda::stream_ref stream) const
{
  return impl_.serialize_updatable(stream);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
hll_sketch<Key, MR, Scope> hll_sketch<Key, MR, Scope>::deserialize(
  ::cuda::stream_ref stream, ::cuda::std::span<const std::uint8_t> bytes, MR mr)
{
  const auto pf = detail::hll::sketch_impl<Key, MR, Scope>::parse_and_validate(bytes);
  hll_sketch sketch(stream, std::move(mr), pf.lgK, HLL_8);
  sketch.impl_.load_registers(stream, bytes);
  return sketch;
}

}  // namespace datasketches::cuda
