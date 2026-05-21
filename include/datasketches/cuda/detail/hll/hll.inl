namespace datasketches::cuda {

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

// Owned stream on device 0.
template <class Key, class MR, ::cuda::thread_scope Scope>
hll_sketch<Key, MR, Scope>::hll_sketch(std::uint8_t lgK, target_hll_type tgt, MR mr)
  : impl_(lgK, tgt, std::move(mr))
{
}

// Borrowed stream.
template <class Key, class MR, ::cuda::thread_scope Scope>
hll_sketch<Key, MR, Scope>::hll_sketch(std::uint8_t lgK,
                                       target_hll_type tgt,
                                       MR mr,
                                       ::cuda::stream_ref stream)
  : impl_(lgK, tgt, std::move(mr), stream)
{
}

// ---------------------------------------------------------------------------
// Stream / update
// ---------------------------------------------------------------------------

template <class Key, class MR, ::cuda::thread_scope Scope>
::cuda::stream_ref hll_sketch<Key, MR, Scope>::stream() const noexcept
{
  return impl_.stream();
}

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class InputIt>
void hll_sketch<Key, MR, Scope>::update(InputIt first, InputIt last)
{
  impl_.update(first, last);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class InputIt>
void hll_sketch<Key, MR, Scope>::update(InputIt first, InputIt last, ::cuda::stream_ref stream)
{
  impl_.update(first, last, stream);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class InputIt>
void hll_sketch<Key, MR, Scope>::update_async(InputIt first, InputIt last)
{
  impl_.update_async(first, last);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class InputIt>
void hll_sketch<Key, MR, Scope>::update_async(InputIt first,
                                              InputIt last,
                                              ::cuda::stream_ref stream)
{
  impl_.update_async(first, last, stream);
}

// ---------------------------------------------------------------------------
// Estimate and bounds
// ---------------------------------------------------------------------------

template <class Key, class MR, ::cuda::thread_scope Scope>
double hll_sketch<Key, MR, Scope>::get_estimate() const
{
  return impl_.get_estimate();
}

template <class Key, class MR, ::cuda::thread_scope Scope>
double hll_sketch<Key, MR, Scope>::get_estimate(::cuda::stream_ref stream) const
{
  return impl_.get_estimate(stream);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
double hll_sketch<Key, MR, Scope>::get_lower_bound(std::uint8_t numStdDev) const
{
  return impl_.get_lower_bound(numStdDev);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
double hll_sketch<Key, MR, Scope>::get_lower_bound(std::uint8_t numStdDev,
                                                   ::cuda::stream_ref stream) const
{
  return impl_.get_lower_bound(numStdDev, stream);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
double hll_sketch<Key, MR, Scope>::get_upper_bound(std::uint8_t numStdDev) const
{
  return impl_.get_upper_bound(numStdDev);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
double hll_sketch<Key, MR, Scope>::get_upper_bound(std::uint8_t numStdDev,
                                                   ::cuda::stream_ref stream) const
{
  return impl_.get_upper_bound(numStdDev, stream);
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class OtherMR, ::cuda::thread_scope OtherScope>
void hll_sketch<Key, MR, Scope>::merge(const hll_sketch<Key, OtherMR, OtherScope>& other)
{
  impl_.merge(other.impl_);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class OtherMR, ::cuda::thread_scope OtherScope>
void hll_sketch<Key, MR, Scope>::merge(const hll_sketch<Key, OtherMR, OtherScope>& other,
                                       ::cuda::stream_ref stream)
{
  impl_.merge(other.impl_, stream);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class OtherMR, ::cuda::thread_scope OtherScope>
void hll_sketch<Key, MR, Scope>::merge_async(const hll_sketch<Key, OtherMR, OtherScope>& other)
{
  impl_.merge_async(other.impl_);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
template <class OtherMR, ::cuda::thread_scope OtherScope>
void hll_sketch<Key, MR, Scope>::merge_async(const hll_sketch<Key, OtherMR, OtherScope>& other,
                                             ::cuda::stream_ref stream)
{
  impl_.merge_async(other.impl_, stream);
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
bool hll_sketch<Key, MR, Scope>::is_empty() const
{
  return impl_.is_empty();
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
std::vector<std::uint8_t> hll_sketch<Key, MR, Scope>::serialize_compact() const
{
  return impl_.serialize_compact();
}

template <class Key, class MR, ::cuda::thread_scope Scope>
std::vector<std::uint8_t> hll_sketch<Key, MR, Scope>::serialize_compact(
  ::cuda::stream_ref stream) const
{
  return impl_.serialize_compact(stream);
}

template <class Key, class MR, ::cuda::thread_scope Scope>
std::vector<std::uint8_t> hll_sketch<Key, MR, Scope>::serialize_updatable() const
{
  return impl_.serialize_updatable();
}

template <class Key, class MR, ::cuda::thread_scope Scope>
std::vector<std::uint8_t> hll_sketch<Key, MR, Scope>::serialize_updatable(
  ::cuda::stream_ref stream) const
{
  return impl_.serialize_updatable(stream);
}

// Owned stream on device 0.
template <class Key, class MR, ::cuda::thread_scope Scope>
hll_sketch<Key, MR, Scope> hll_sketch<Key, MR, Scope>::deserialize(
  ::cuda::std::span<const std::uint8_t> bytes, MR mr)
{
  const auto pf = detail::hll_sketch_impl<Key, MR, Scope>::parse_and_validate(bytes);
  hll_sketch sketch(pf.lgK, HLL_8, std::move(mr));
  sketch.impl_.load_registers(bytes);
  return sketch;
}

// Borrowed stream.
template <class Key, class MR, ::cuda::thread_scope Scope>
hll_sketch<Key, MR, Scope> hll_sketch<Key, MR, Scope>::deserialize(
  ::cuda::std::span<const std::uint8_t> bytes, MR mr, ::cuda::stream_ref stream)
{
  const auto pf = detail::hll_sketch_impl<Key, MR, Scope>::parse_and_validate(bytes);
  hll_sketch sketch(pf.lgK, HLL_8, std::move(mr), stream);
  sketch.impl_.load_registers(bytes);
  return sketch;
}

}  // namespace datasketches::cuda
