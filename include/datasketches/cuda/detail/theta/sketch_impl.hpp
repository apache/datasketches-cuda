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
#include <cstddef>
#include <cstdint>
#include <cuda/memory_pool>
#include <cuda/std/bit>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/stream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <cub/device/device_merge.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/execution.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <datasketches/cuda/detail/common/error.hpp>
#include <datasketches/cuda/detail/theta/policy.cuh>
#include <datasketches/cuda/detail/theta/preamble.hpp>
#include <datasketches/cuda/detail/theta/screen.cuh>

namespace datasketches::cuda::detail::theta {

template <class Key, class MR = ::cuda::device_memory_pool_ref>
struct sketch_impl {
  using hash_result_type  = std::uint64_t;
  using count_type        = std::uint64_t;
  using hash_buffer_type  = ::cuda::device_buffer<hash_result_type>;
  using count_buffer_type = ::cuda::device_buffer<count_type>;
  using env_type          = ::cuda::experimental::env_t<::cuda::mr::device_accessible>;

  struct buffer_result {
    hash_buffer_type data;
    std::size_t size;
  };

  std::uint8_t lg_k_;
  std::uint64_t seed_;
  std::uint64_t initial_theta_;
  std::uint64_t theta_;
  bool is_empty_;
  std::size_t survivor_budget_;  // Screen output capacity in hashes; tests lower it.
  ::cuda::stream_ref allocation_stream_;
  MR mr_;
  hash_buffer_type hashes_;

  sketch_impl(::cuda::stream_ref stream, MR mr, std::uint8_t lg_k, std::uint64_t seed, float p)
    : lg_k_(check_lg_k(lg_k)),
      seed_(seed),
      initial_theta_(checked_theta(p)),
      theta_(initial_theta_),
      is_empty_(true),
      survivor_budget_(std::max(tuning::min_survivor_budget, tuning::chunk_target_multiple * k())),
      allocation_stream_(stream),
      mr_(std::move(mr)),
      hashes_(make_hash_buffer(stream, 0))
  {
  }

  sketch_impl(const sketch_impl&)            = delete;
  sketch_impl& operator=(const sketch_impl&) = delete;
  sketch_impl(sketch_impl&&)                 = default;
  sketch_impl& operator=(sketch_impl&&)      = default;

  static std::uint8_t check_lg_k(std::uint8_t lg_k)
  {
    if (lg_k < min_lg_k || lg_k > max_lg_k) {
      throw std::invalid_argument("theta_sketch lg_k must be in [5, 26]");
    }
    return lg_k;
  }

  static std::uint64_t checked_theta(float p)
  {
    const auto theta = p > 0.0F && p < 1.0F
                         ? static_cast<std::uint64_t>(static_cast<double>(max_theta) * p)
                         : max_theta;
    if (!(p > 0.0F && p <= 1.0F) || theta == 0) {
      throw std::invalid_argument(
        "theta_sketch sampling probability must be in (0, 1] and yield a nonzero theta");
    }
    return theta;
  }

  [[nodiscard]] std::size_t k() const noexcept { return std::size_t{1} << lg_k_; }

  [[nodiscard]] std::uint64_t effective_theta() const noexcept
  {
    return is_empty_ ? max_theta : theta_;
  }

  [[nodiscard]] double theta_rate() const noexcept
  {
    return static_cast<double>(theta_) / static_cast<double>(max_theta);
  }

  [[nodiscard]] env_type env(::cuda::stream_ref stream) const { return env_type{mr_, stream}; }

  [[nodiscard]] hash_buffer_type make_hash_buffer(::cuda::stream_ref stream,
                                                  std::size_t count) const
  {
    return ::cuda::make_buffer<hash_result_type, ::cuda::mr::device_accessible>(
      stream, mr_, count, ::cuda::no_init);
  }

  [[nodiscard]] count_buffer_type make_count_buffer(::cuda::stream_ref stream) const
  {
    return ::cuda::make_buffer<count_type, ::cuda::mr::device_accessible>(
      stream, mr_, 1, count_type{0});
  }

  [[nodiscard]] static std::size_t read_count(::cuda::stream_ref stream,
                                              const count_buffer_type& count)
  {
    count_type host_count{};
    DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(
      &host_count, count.data(), sizeof(host_count), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    return static_cast<std::size_t>(host_count);
  }

  template <class Predicate>
  [[nodiscard]] buffer_result select(::cuda::stream_ref stream,
                                     const hash_result_type* input,
                                     std::size_t count,
                                     Predicate predicate) const
  {
    auto output = make_hash_buffer(stream, count);
    if (count == 0) return {std::move(output), 0};
    auto selected = make_count_buffer(stream);
    DATASKETCHES_CUDA_TRY(cub::DeviceSelect::If(input,
                                                output.data(),
                                                selected.data(),
                                                static_cast<std::int64_t>(count),
                                                predicate,
                                                env(stream)));
    return {std::move(output), read_count(stream, selected)};
  }

  [[nodiscard]] buffer_result unique_sorted(::cuda::stream_ref stream,
                                            const hash_result_type* sorted,
                                            std::size_t count) const
  {
    auto output   = make_hash_buffer(stream, count);
    auto selected = make_count_buffer(stream);
    DATASKETCHES_CUDA_TRY(cub::DeviceSelect::Unique(
      sorted, output.data(), selected.data(), static_cast<std::int64_t>(count), env(stream)));
    return {std::move(output), read_count(stream, selected)};
  }

  [[nodiscard]] buffer_result sort_unique(::cuda::stream_ref stream,
                                          hash_buffer_type input,
                                          std::size_t count,
                                          std::uint64_t bound) const
  {
    auto alternate = make_hash_buffer(stream, count);
    cub::DoubleBuffer<hash_result_type> keys(input.data(), alternate.data());
    DATASKETCHES_CUDA_TRY(cub::DeviceRadixSort::SortKeys(
      keys, static_cast<std::int64_t>(count), 0, ::cuda::std::bit_width(bound), env(stream)));
    return unique_sorted(stream, keys.Current(), count);
  }

  // Count every survivor but bound writes; an oversized result must be retried.
  template <class RandomAccessIt>
  [[nodiscard]] buffer_result screen(::cuda::stream_ref stream,
                                     RandomAccessIt first,
                                     std::size_t count) const
  {
    const std::size_t capacity = std::min(count, survivor_budget_);
    auto output                = make_hash_buffer(stream, capacity);
    auto selected              = make_count_buffer(stream);
    screen_kernel<<<tuning::grid_size(count), tuning::block_threads, 0, stream.get()>>>(
      first, count, theta_hash<Key>{seed_}, theta_, output.data(), capacity, selected.data());
    DATASKETCHES_CUDA_TRY(cudaGetLastError());
    return {std::move(output), read_count(stream, selected)};
  }

  //! @brief Merge sorted, individually unique ranges; only cross-range duplicates remain.
  [[nodiscard]] buffer_result merge_unique(::cuda::stream_ref stream,
                                           const hash_result_type* first,
                                           std::size_t first_size,
                                           const hash_result_type* second,
                                           std::size_t second_size) const
  {
    const std::size_t merged_size = first_size + second_size;
    auto merged                   = make_hash_buffer(stream, merged_size);
    if (merged_size == 0) return {std::move(merged), 0};

    if (first_size == 0 || second_size == 0) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(merged.data(),
                                            first_size == 0 ? second : first,
                                            merged_size * sizeof(hash_result_type),
                                            cudaMemcpyDeviceToDevice,
                                            stream.get()));
      return {std::move(merged), merged_size};
    }
    DATASKETCHES_CUDA_TRY(cub::DeviceMerge::MergeKeys(first,
                                                      static_cast<std::int64_t>(first_size),
                                                      second,
                                                      static_cast<std::int64_t>(second_size),
                                                      merged.data(),
                                                      ::cuda::std::less<>{},
                                                      env(stream)));
    return unique_sorted(stream, merged.data(), merged_size);
  }

  void install(::cuda::stream_ref stream, buffer_result input, std::uint64_t theta, bool empty)
  {
    const auto retained = std::min(input.size, k());
    const auto* data    = input.data.data();
    const bool reuse    = input.data.size() == retained;
    auto next           = reuse ? std::move(input.data) : make_hash_buffer(stream, retained);
    if (!reuse && retained != 0) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(next.data(),
                                            data,
                                            retained * sizeof(hash_result_type),
                                            cudaMemcpyDeviceToDevice,
                                            stream.get()));
    }
    if (input.size > k()) {
      DATASKETCHES_CUDA_TRY(
        cudaMemcpyAsync(&theta, data + k(), sizeof(theta), cudaMemcpyDeviceToHost, stream.get()));
    }
    stream.sync();
    next.set_stream(allocation_stream_);
    hashes_   = std::move(next);
    theta_    = empty ? initial_theta_ : theta;
    is_empty_ = empty;
  }

  // Predict survivors from theta and the last chunk's duplicate rate. Hedge
  // that rate and cap growth because the distribution can change mid-update.
  // At maximum theta, double chunks so distinct inputs tighten theta early.
  // Guarded writes and halving on overflow guarantee progress for any input.
  [[nodiscard]] std::size_t next_chunk(std::size_t remaining,
                                       std::size_t previous,
                                       double duplicate_factor) const noexcept
  {
    const double hedged = std::min(1.0, tuning::duplicate_hedge * duplicate_factor);
    const double rate   = theta_rate() * hedged;
    const double target = rate > 0.0
                            ? static_cast<double>(survivor_budget_) / (tuning::budget_margin * rate)
                            : static_cast<double>(remaining);
    std::size_t sized =
      target >= static_cast<double>(remaining) ? remaining : static_cast<std::size_t>(target);
    if (hedged < 1.0 && previous != 0 && sized / tuning::max_chunk_growth > previous) {
      sized = previous * tuning::max_chunk_growth;
    }
    sized = std::max<std::size_t>(sized, 1);
    if (theta_ == max_theta) {
      const auto doubling =
        std::max({tuning::chunk_target_multiple * k(), tuning::min_chunk_keys, previous * 2});
      return std::min({doubling, sized, remaining});
    }
    return std::min(std::max(sized, tuning::min_chunk_keys), remaining);
  }

  template <class RandomAccessIt>
  [[nodiscard]] std::size_t update_chunk(::cuda::stream_ref stream,
                                         RandomAccessIt first,
                                         std::size_t count)
  {
    auto screened = screen(stream, first, count);
    if (screened.size > screened.data.size() || screened.size == 0) return screened.size;

    auto incoming = sort_unique(stream, std::move(screened.data), screened.size, theta_);
    auto combined =
      hashes_.empty()
        ? std::move(incoming)
        : merge_unique(stream, hashes_.data(), hashes_.size(), incoming.data.data(), incoming.size);
    install(stream, std::move(combined), theta_, false);
    return screened.size;
  }

  template <class RandomAccessIt>
  void update(::cuda::stream_ref stream, RandomAccessIt first, RandomAccessIt last)
  {
    const auto distance = last - first;
    if (distance < 0) {
      throw std::invalid_argument("theta_sketch::update requires a non-negative range");
    }
    const auto count = static_cast<std::size_t>(distance);

    std::size_t offset      = 0;
    std::size_t previous    = 0;
    double duplicate_factor = 1.0;
    while (offset < count) {
      auto chunk = next_chunk(count - offset, previous, duplicate_factor);
      // Theta is fixed until something is installed, so a pass's survivors per
      // key over the theta fraction it was screened at is its duplicate factor.
      const double rate = theta_rate();
      auto outcome      = update_chunk(stream, first + offset, chunk);
      while (outcome > survivor_budget_) {
        duplicate_factor = static_cast<double>(outcome) / static_cast<double>(chunk) / rate;
        // The exact survivor count of the failed pass usually lands the retry
        // within the budget at once; halving guarantees progress regardless,
        // since a chunk of at most the budget cannot overflow.
        chunk   = std::min(chunk / 2, next_chunk(count - offset, previous, duplicate_factor));
        outcome = update_chunk(stream, first + offset, chunk);
      }
      // Commit the empty flag only after a chunk succeeds. A chunk whose keys
      // are all rejected still counts as seen, but a failed first allocation
      // must leave an empty sketch unchanged.
      is_empty_        = false;
      duplicate_factor = static_cast<double>(outcome) / static_cast<double>(chunk) / rate;
      offset += chunk;
      previous = chunk;
    }
  }

  template <class OtherMR>
  void merge(::cuda::stream_ref stream, const sketch_impl<Key, OtherMR>& other)
  {
    if (other.is_empty_) return;
    if (::compute_seed_hash(seed_) != ::compute_seed_hash(other.seed_)) {
      throw std::invalid_argument("theta_sketch::merge: seed hash mismatch");
    }

    const std::uint64_t theta = std::min(theta_, other.effective_theta());
    auto combined             = merge_unique(
      stream, hashes_.data(), hashes_.size(), other.hashes_.data(), other.hashes_.size());
    auto screened = select(stream, combined.data.data(), combined.size, screen_hash{theta});
    install(stream, std::move(screened), theta, false);
  }

  template <class OtherMR>
  void intersect(::cuda::stream_ref stream, const sketch_impl<Key, OtherMR>& other)
  {
    if (is_empty_) return;
    if (other.is_empty_) {
      reset(stream);
      return;
    }
    retain_matching(stream, other, true);
  }

  template <class OtherMR>
  void a_not_b(::cuda::stream_ref stream, const sketch_impl<Key, OtherMR>& other)
  {
    if (is_empty_ || (!hashes_.empty() && other.is_empty_)) return;
    retain_matching(stream, other, false);
  }

  template <class OtherMR>
  void retain_matching(::cuda::stream_ref stream,
                       const sketch_impl<Key, OtherMR>& other,
                       bool keep_matches)
  {
    if (::compute_seed_hash(seed_) != ::compute_seed_hash(other.seed_)) {
      throw std::invalid_argument(keep_matches ? "theta_sketch::intersect: seed hash mismatch"
                                               : "theta_sketch::a_not_b: seed hash mismatch");
    }
    const auto theta = std::min(effective_theta(), other.effective_theta());
    auto result =
      select(stream,
             hashes_.data(),
             hashes_.size(),
             membership_filter{other.hashes_.data(), other.hashes_.size(), theta, keep_matches});
    const bool empty = result.size == 0 && theta == max_theta;
    install(stream, std::move(result), theta, empty);
  }

  void reset(::cuda::stream_ref stream)
  {
    install(stream, {make_hash_buffer(stream, 0), 0}, max_theta, true);
  }

  [[nodiscard]] std::vector<hash_result_type> get_retained_hashes(::cuda::stream_ref stream) const
  {
    std::vector<hash_result_type> entries(hashes_.size());
    if (!entries.empty()) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(entries.data(),
                                            hashes_.data(),
                                            entries.size() * sizeof(hash_result_type),
                                            cudaMemcpyDeviceToHost,
                                            stream.get()));
    }
    stream.sync();
    return entries;
  }

  [[nodiscard]] std::vector<std::uint8_t> serialize_compact(::cuda::stream_ref stream) const
  {
    return serialize_compact_v3(
      is_empty_, ::compute_seed_hash(seed_), effective_theta(), get_retained_hashes(stream));
  }

  void load_compact(::cuda::stream_ref stream, const compact_image& image)
  {
    if (image.entries.size() > k()) {
      throw std::invalid_argument(
        "theta_sketch::deserialize: retained entries exceed configured nominal k");
    }
    auto next = make_hash_buffer(stream, image.entries.size());
    if (!image.entries.empty()) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(next.data(),
                                            image.entries.data(),
                                            image.entries.size() * sizeof(hash_result_type),
                                            cudaMemcpyHostToDevice,
                                            stream.get()));
    }
    install(stream, {std::move(next), image.entries.size()}, image.theta, image.empty);
  }
};

}  // namespace datasketches::cuda::detail::theta
