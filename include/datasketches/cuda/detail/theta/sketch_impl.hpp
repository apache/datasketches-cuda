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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda/memory_pool>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/stream>
#include <iterator>
#include <limits>
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
#include <datasketches/cuda/detail/theta/persistent_update.cuh>
#include <datasketches/cuda/detail/theta/policy.cuh>
#include <datasketches/cuda/detail/theta/preamble.hpp>

#include <binomial_bounds.hpp>

namespace datasketches::cuda::detail::theta {

template <class Key, class MR = ::cuda::device_memory_pool_ref>
struct sketch_impl {
  using key_type          = Key;
  using hash_type         = std::uint64_t;
  using count_type        = std::uint64_t;
  using hash_buffer_type  = ::cuda::device_buffer<hash_type>;
  using count_buffer_type = ::cuda::device_buffer<count_type>;
  using state_buffer_type = ::cuda::device_buffer<device_state>;
  using env_type          = ::cuda::experimental::env_t<::cuda::mr::device_accessible>;

  struct buffer_result {
    hash_buffer_type data;
    std::size_t size;
  };

  struct persistent_runtime_config {
    std::size_t set_slots;
    std::size_t shared_bytes;
    std::size_t sm_count;
    std::size_t max_grid_blocks;
  };

  [[nodiscard]] static persistent_runtime_config make_persistent_runtime_config_()
  {
    int device{};
    int max_block_shared{};
    int max_sm_shared{};
    int sms{};
    DATASKETCHES_CUDA_TRY(cudaGetDevice(&device));
    DATASKETCHES_CUDA_TRY(
      cudaDeviceGetAttribute(&max_block_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    DATASKETCHES_CUDA_TRY(
      cudaDeviceGetAttribute(&max_sm_shared, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    DATASKETCHES_CUDA_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));

    if (max_block_shared <= 0 || max_sm_shared <= 0 || sms <= 0) {
      throw std::runtime_error("theta_sketch could not determine device resources");
    }

    const auto fixed_bytes =
      histogram_bytes + sizeof(persistent_control) + alignof(persistent_control);
    const auto shared_budget = std::min(
      static_cast<std::size_t>(max_block_shared),
      static_cast<std::size_t>(max_sm_shared) / persistent_update_config::target_blocks_per_sm);
    if (shared_budget <= fixed_bytes) {
      throw std::runtime_error("theta_sketch device has insufficient shared memory");
    }

    const auto slots = (shared_budget - fixed_bytes) / sizeof(hash_type);
    const auto bytes = theta::required_shared_bytes(slots);
    if (max_occupied(slots) <
        persistent_update_config::retained_capacity + persistent_update_config::max_block_threads) {
      throw std::runtime_error("theta_sketch device cannot fit k plus one block in shared memory");
    }
    const auto blocks_per_sm = std::max(1, max_sm_shared / static_cast<int>(bytes));
    return {slots,
            bytes,
            static_cast<std::size_t>(sms),
            static_cast<std::size_t>(sms) * static_cast<std::size_t>(blocks_per_sm)};
  }

  std::uint8_t lg_k_;
  std::uint64_t seed_;
  float p_;
  MR mr_;
  persistent_runtime_config persistent_config_;
  std::size_t candidate_capacity_;
  hash_buffer_type hashes_;
  state_buffer_type state_;
  hash_buffer_type candidates_;
  hash_buffer_type alternate_;
  hash_buffer_type unique_;
  count_buffer_type unique_count_;

  sketch_impl(::cuda::stream_ref stream, MR mr, std::uint8_t lg_k, std::uint64_t seed, float p)
    : lg_k_(check_lg_k_(lg_k)),
      seed_(seed),
      p_(check_p_(p)),
      mr_(std::move(mr)),
      persistent_config_(make_persistent_runtime_config_()),
      candidate_capacity_(k_() + persistent_config_.max_grid_blocks *
                                   persistent_update_config::block_output_slots),
      hashes_(make_filled_hash_buffer_(stream, k_(), empty_key)),
      state_(make_state_buffer_(stream, device_state{starting_theta_(p_), 0, 1})),
      candidates_(make_hash_buffer_(stream, candidate_capacity_)),
      alternate_(make_hash_buffer_(stream, candidate_capacity_)),
      unique_(make_hash_buffer_(stream, candidate_capacity_)),
      unique_count_(make_count_buffer_(stream))
  {
  }

  sketch_impl(const sketch_impl&)            = delete;
  sketch_impl& operator=(const sketch_impl&) = delete;
  sketch_impl(sketch_impl&&)                 = default;
  sketch_impl& operator=(sketch_impl&&)      = default;
  ~sketch_impl()                             = default;

  static std::uint8_t check_lg_k_(std::uint8_t lg_k)
  {
    if (lg_k != default_lg_k) {
      throw std::invalid_argument("theta_sketch prototype supports only lg_k == 12");
    }
    return lg_k;
  }

  static float check_p_(float p)
  {
    if (!(p > 0.0F && p <= 1.0F)) {
      throw std::invalid_argument("theta_sketch sampling probability must be in (0, 1]");
    }
    return p;
  }

  static std::uint64_t starting_theta_(float p)
  {
    return p < 1.0F ? static_cast<std::uint64_t>(static_cast<double>(max_theta) * p) : max_theta;
  }

  [[nodiscard]] std::size_t k_() const noexcept { return std::size_t{1} << lg_k_; }

  [[nodiscard]] env_type env_(::cuda::stream_ref stream) const { return env_type{mr_, stream}; }

  [[nodiscard]] hash_buffer_type make_hash_buffer_(::cuda::stream_ref stream,
                                                   std::size_t count) const
  {
    return ::cuda::make_buffer<hash_type, ::cuda::mr::device_accessible>(
      stream, mr_, count, ::cuda::no_init);
  }

  [[nodiscard]] hash_buffer_type make_filled_hash_buffer_(::cuda::stream_ref stream,
                                                          std::size_t count,
                                                          hash_type value) const
  {
    return ::cuda::make_buffer<hash_type, ::cuda::mr::device_accessible>(stream, mr_, count, value);
  }

  [[nodiscard]] count_buffer_type make_count_buffer_(::cuda::stream_ref stream) const
  {
    return ::cuda::make_buffer<count_type, ::cuda::mr::device_accessible>(
      stream, mr_, 1, count_type{0});
  }

  [[nodiscard]] state_buffer_type make_state_buffer_(::cuda::stream_ref stream,
                                                     device_state state) const
  {
    return ::cuda::make_buffer<device_state, ::cuda::mr::device_accessible>(stream, mr_, 1, state);
  }

  [[nodiscard]] static std::int64_t cub_count_(std::size_t count)
  {
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::length_error("theta_sketch input exceeds CUB's signed 64-bit item count");
    }
    return static_cast<std::int64_t>(count);
  }

  [[nodiscard]] static std::size_t read_count_(::cuda::stream_ref stream,
                                               const count_buffer_type& count)
  {
    count_type host_count{};
    DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(
      &host_count, count.data(), sizeof(host_count), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    if (host_count > std::numeric_limits<std::size_t>::max()) {
      throw std::length_error("theta_sketch selected item count exceeds size_t");
    }
    return static_cast<std::size_t>(host_count);
  }

  [[nodiscard]] device_state read_state_(::cuda::stream_ref stream) const
  {
    device_state state{};
    DATASKETCHES_CUDA_TRY(
      cudaMemcpyAsync(&state, state_.data(), sizeof(state), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    return state;
  }

  void write_state_(::cuda::stream_ref stream, device_state state)
  {
    DATASKETCHES_CUDA_TRY(
      cudaMemcpyAsync(state_.data(), &state, sizeof(state), cudaMemcpyHostToDevice, stream.get()));
  }

  [[nodiscard]] static std::uint64_t effective_theta_(device_state state) noexcept
  {
    return state.empty != 0 ? max_theta : state.theta;
  }

  struct persistent_launch_config {
    int grid_size;
    int block_size;
  };

  template <class RandomAccessIt>
  [[nodiscard]] persistent_launch_config configure_persistent_kernel_() const
  {
    const auto kernel = persistent_update_kernel<RandomAccessIt, theta_hash<Key>>;
    DATASKETCHES_CUDA_TRY(cudaFuncSetAttribute(kernel,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(persistent_config_.shared_bytes)));

    int minimum_grid_size{};
    int block_size{};
    DATASKETCHES_CUDA_TRY(
      cudaOccupancyMaxPotentialBlockSize(&minimum_grid_size,
                                         &block_size,
                                         kernel,
                                         persistent_config_.shared_bytes,
                                         persistent_update_config::max_block_threads));

    const auto safe_threads =
      max_occupied(persistent_config_.set_slots) - persistent_update_config::retained_capacity;
    block_size = std::min(block_size, static_cast<int>(safe_threads));
    block_size = block_size / 32 * 32;
    if (block_size != persistent_update_config::max_block_threads) {
      throw std::runtime_error("theta_sketch persistent kernel requires a 1024-thread block");
    }

    int active_blocks{};
    DATASKETCHES_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, block_size, persistent_config_.shared_bytes));
    if (active_blocks < 1) {
      throw std::runtime_error(
        "theta_sketch persistent kernel has no resident launch configuration");
    }
    const auto grid_size = static_cast<std::size_t>(active_blocks) * persistent_config_.sm_count;
    if (grid_size > persistent_config_.max_grid_blocks) {
      throw std::runtime_error("theta_sketch persistent grid exceeds scratch capacity");
    }
    return {static_cast<int>(grid_size), block_size};
  }

  [[nodiscard]] buffer_result select_(::cuda::stream_ref stream,
                                      const hash_type* input,
                                      std::size_t count,
                                      screen_hash predicate) const
  {
    auto output = make_hash_buffer_(stream, count);
    if (count == 0) return {std::move(output), 0};
    auto selected = make_count_buffer_(stream);
    DATASKETCHES_CUDA_TRY(cub::DeviceSelect::If(
      input, output.data(), selected.data(), cub_count_(count), predicate, env_(stream)));
    return {std::move(output), read_count_(stream, selected)};
  }

  [[nodiscard]] buffer_result select_(::cuda::stream_ref stream,
                                      const hash_type* input,
                                      std::size_t count,
                                      membership_filter predicate) const
  {
    auto output = make_hash_buffer_(stream, count);
    if (count == 0) return {std::move(output), 0};
    auto selected = make_count_buffer_(stream);
    DATASKETCHES_CUDA_TRY(cub::DeviceSelect::If(
      input, output.data(), selected.data(), cub_count_(count), predicate, env_(stream)));
    return {std::move(output), read_count_(stream, selected)};
  }

  [[nodiscard]] buffer_result unique_sorted_(::cuda::stream_ref stream,
                                             const hash_type* sorted,
                                             std::size_t count) const
  {
    auto output = make_hash_buffer_(stream, count);
    if (count == 0) return {std::move(output), 0};
    auto selected = make_count_buffer_(stream);
    DATASKETCHES_CUDA_TRY(cub::DeviceSelect::Unique(
      sorted, output.data(), selected.data(), cub_count_(count), env_(stream)));
    return {std::move(output), read_count_(stream, selected)};
  }

  //! @brief Highest bit position that a hash below @p theta can occupy.
  //!
  //! Radix sort runs one pass per fixed number of bits, so bounding the key range
  //! by theta drops whole passes once the sketch has left the initial
  //! theta == max_theta state.
  [[nodiscard]] static int significant_bits_(std::uint64_t theta) noexcept
  {
    int bits = 0;
    while (theta != 0) {
      ++bits;
      theta >>= 1;
    }
    return bits == 0 ? 1 : bits;
  }

  [[nodiscard]] buffer_result sort_unique_(::cuda::stream_ref stream,
                                           hash_buffer_type&& input,
                                           std::size_t count,
                                           std::uint64_t bound) const
  {
    if (count == 0) return {std::move(input), 0};
    auto alternate = make_hash_buffer_(stream, count);
    cub::DoubleBuffer<hash_type> keys(input.data(), alternate.data());
    DATASKETCHES_CUDA_TRY(cub::DeviceRadixSort::SortKeys(
      keys, cub_count_(count), 0, significant_bits_(bound), env_(stream)));
    return unique_sorted_(stream, keys.Current(), count);
  }

  [[nodiscard]] buffer_result merge_unique_(::cuda::stream_ref stream,
                                            const hash_type* first,
                                            std::size_t first_size,
                                            const hash_type* second,
                                            std::size_t second_size) const
  {
    const std::size_t merged_size = first_size + second_size;
    auto merged                   = make_hash_buffer_(stream, merged_size);
    if (merged_size == 0) return {std::move(merged), 0};

    if (first_size == 0) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(merged.data(),
                                            second,
                                            second_size * sizeof(hash_type),
                                            cudaMemcpyDeviceToDevice,
                                            stream.get()));
    } else if (second_size == 0) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(merged.data(),
                                            first,
                                            first_size * sizeof(hash_type),
                                            cudaMemcpyDeviceToDevice,
                                            stream.get()));
    } else {
      DATASKETCHES_CUDA_TRY(cub::DeviceMerge::MergeKeys(first,
                                                        cub_count_(first_size),
                                                        second,
                                                        cub_count_(second_size),
                                                        merged.data(),
                                                        ::cuda::std::less<>{},
                                                        env_(stream)));
    }
    return unique_sorted_(stream, merged.data(), merged_size);
  }

  void install_(::cuda::stream_ref stream,
                const hash_type* input,
                std::size_t count,
                std::uint64_t theta,
                bool empty,
                bool trim)
  {
    std::size_t retained = count;
    if (trim && count > k_()) {
      DATASKETCHES_CUDA_TRY(
        cudaMemcpyAsync(&theta, input + k_(), sizeof(theta), cudaMemcpyDeviceToHost, stream.get()));
      stream.sync();
      retained = k_();
    }
    DATASKETCHES_CUDA_TRY(
      cudaMemsetAsync(hashes_.data(), 0xff, hashes_.size() * sizeof(hash_type), stream.get()));
    if (retained != 0) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(hashes_.data(),
                                            input,
                                            retained * sizeof(hash_type),
                                            cudaMemcpyDeviceToDevice,
                                            stream.get()));
    }
    write_state_(stream,
                 device_state{empty ? starting_theta_(p_) : theta, retained, empty ? 1U : 0U});
    stream.sync();
  }

  void set_empty_(::cuda::stream_ref stream)
  {
    DATASKETCHES_CUDA_TRY(
      cudaMemsetAsync(hashes_.data(), 0xff, hashes_.size() * sizeof(hash_type), stream.get()));
    write_state_(stream, device_state{starting_theta_(p_), 0, 1});
    stream.sync();
  }

  template <class RandomAccessIt>
  void update_async(::cuda::stream_ref stream, RandomAccessIt first, RandomAccessIt last)
  {
    const auto distance = last - first;
    if (distance < 0) {
      throw std::invalid_argument("theta_sketch::update requires a non-negative range");
    }
    const auto count = static_cast<std::size_t>(distance);
    if (count == 0) return;
    const auto launch          = configure_persistent_kernel_<RandomAccessIt>();
    const auto blocks          = std::min(static_cast<std::size_t>(launch.grid_size),
                                 (count + static_cast<std::size_t>(launch.block_size) - 1) /
                                   static_cast<std::size_t>(launch.block_size));
    const auto candidate_count = k_() + blocks * persistent_update_config::block_output_slots;

    DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(candidates_.data(),
                                          hashes_.data(),
                                          k_() * sizeof(hash_type),
                                          cudaMemcpyDeviceToDevice,
                                          stream.get()));

    persistent_update_kernel<<<static_cast<unsigned int>(blocks),
                               launch.block_size,
                               persistent_config_.shared_bytes,
                               stream.get()>>>(first,
                                               count,
                                               theta_hash<Key>{seed_},
                                               state_.data(),
                                               candidates_.data() + k_(),
                                               persistent_config_.set_slots,
                                               max_occupied(persistent_config_.set_slots));
    DATASKETCHES_CUDA_TRY(cudaGetLastError());

    cub::DoubleBuffer<hash_type> sorted(candidates_.data(), alternate_.data());
    DATASKETCHES_CUDA_TRY(
      cub::DeviceRadixSort::SortKeys(sorted, cub_count_(candidate_count), 0, 63, env_(stream)));
    DATASKETCHES_CUDA_TRY(cub::DeviceSelect::Unique(sorted.Current(),
                                                    unique_.data(),
                                                    unique_count_.data(),
                                                    cub_count_(candidate_count),
                                                    env_(stream)));

    finalize_update_kernel<<<1, 256, 0, stream.get()>>>(
      unique_.data(), unique_count_.data(), hashes_.data(), state_.data(), true);
    DATASKETCHES_CUDA_TRY(cudaGetLastError());
  }

  template <class RandomAccessIt>
  void update(::cuda::stream_ref stream, RandomAccessIt first, RandomAccessIt last)
  {
    update_async(stream, first, last);
    stream.sync();
  }

  template <class OtherMR>
  void merge(::cuda::stream_ref stream, const sketch_impl<Key, OtherMR>& other)
  {
    const auto self_state  = read_state_(stream);
    const auto other_state = other.read_state_(stream);
    if (other_state.empty != 0) return;
    if (::compute_seed_hash(seed_) != ::compute_seed_hash(other.seed_)) {
      throw std::invalid_argument("theta_sketch::merge: seed hash mismatch");
    }

    const std::uint64_t theta = std::min(self_state.theta, effective_theta_(other_state));
    auto combined             = merge_unique_(stream,
                                  hashes_.data(),
                                  static_cast<std::size_t>(self_state.count),
                                  other.hashes_.data(),
                                  static_cast<std::size_t>(other_state.count));
    auto screened = select_(stream, combined.data.data(), combined.size, screen_hash{theta});
    install_(stream, screened.data.data(), screened.size, theta, false, true);
  }

  template <class OtherMR>
  void intersect(::cuda::stream_ref stream, const sketch_impl<Key, OtherMR>& other)
  {
    const auto self_state  = read_state_(stream);
    const auto other_state = other.read_state_(stream);
    if (self_state.empty != 0) return;
    if (other_state.empty != 0) {
      set_empty_(stream);
      return;
    }
    if (::compute_seed_hash(seed_) != ::compute_seed_hash(other.seed_)) {
      throw std::invalid_argument("theta_sketch::intersect: seed hash mismatch");
    }

    const std::uint64_t theta =
      std::min(effective_theta_(self_state), effective_theta_(other_state));
    auto result =
      select_(stream,
              hashes_.data(),
              static_cast<std::size_t>(self_state.count),
              membership_filter{
                other.hashes_.data(), static_cast<std::size_t>(other_state.count), theta, true});
    const bool empty = result.size == 0 && theta == max_theta;
    install_(stream, result.data.data(), result.size, theta, empty, false);
  }

  template <class OtherMR>
  void a_not_b(::cuda::stream_ref stream, const sketch_impl<Key, OtherMR>& other)
  {
    const auto self_state  = read_state_(stream);
    const auto other_state = other.read_state_(stream);
    if (self_state.empty != 0 || (self_state.count != 0 && other_state.empty != 0)) return;
    if (::compute_seed_hash(seed_) != ::compute_seed_hash(other.seed_)) {
      throw std::invalid_argument("theta_sketch::a_not_b: seed hash mismatch");
    }

    const std::uint64_t theta =
      std::min(effective_theta_(self_state), effective_theta_(other_state));
    auto result =
      select_(stream,
              hashes_.data(),
              static_cast<std::size_t>(self_state.count),
              membership_filter{
                other.hashes_.data(), static_cast<std::size_t>(other_state.count), theta, false});
    const bool empty = result.size == 0 && theta == max_theta;
    install_(stream, result.data.data(), result.size, theta, empty, false);
  }

  void reset(::cuda::stream_ref stream) { set_empty_(stream); }

  [[nodiscard]] bool is_empty(::cuda::stream_ref stream) const
  {
    return read_state_(stream).empty != 0;
  }

  [[nodiscard]] bool is_estimation_mode(::cuda::stream_ref stream) const
  {
    const auto state = read_state_(stream);
    return state.empty == 0 && state.theta < max_theta;
  }

  [[nodiscard]] std::uint8_t get_lg_k() const noexcept { return lg_k_; }

  [[nodiscard]] std::uint64_t get_theta64(::cuda::stream_ref stream) const
  {
    return effective_theta_(read_state_(stream));
  }

  [[nodiscard]] double get_theta(::cuda::stream_ref stream) const
  {
    return static_cast<double>(get_theta64(stream)) / static_cast<double>(max_theta);
  }

  [[nodiscard]] std::uint16_t get_seed_hash() const noexcept { return ::compute_seed_hash(seed_); }

  [[nodiscard]] std::size_t get_num_retained(::cuda::stream_ref stream) const
  {
    return static_cast<std::size_t>(read_state_(stream).count);
  }

  [[nodiscard]] double get_estimate(::cuda::stream_ref stream) const
  {
    const auto state = read_state_(stream);
    return static_cast<double>(state.count) /
           (static_cast<double>(effective_theta_(state)) / static_cast<double>(max_theta));
  }

  [[nodiscard]] double get_lower_bound(::cuda::stream_ref stream, std::uint8_t num_std_devs) const
  {
    const auto state = read_state_(stream);
    if (state.empty != 0 || state.theta == max_theta) { return static_cast<double>(state.count); }
    return ::datasketches::binomial_bounds::get_lower_bound(
      state.count, static_cast<double>(state.theta) / static_cast<double>(max_theta), num_std_devs);
  }

  [[nodiscard]] double get_upper_bound(::cuda::stream_ref stream, std::uint8_t num_std_devs) const
  {
    const auto state = read_state_(stream);
    if (state.empty != 0 || state.theta == max_theta) { return static_cast<double>(state.count); }
    return ::datasketches::binomial_bounds::get_upper_bound(
      state.count, static_cast<double>(state.theta) / static_cast<double>(max_theta), num_std_devs);
  }

  [[nodiscard]] std::vector<hash_type> get_retained_hashes(::cuda::stream_ref stream) const
  {
    device_state state{};
    std::vector<hash_type> entries(k_());
    DATASKETCHES_CUDA_TRY(
      cudaMemcpyAsync(&state, state_.data(), sizeof(state), cudaMemcpyDeviceToHost, stream.get()));
    DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(entries.data(),
                                          hashes_.data(),
                                          entries.size() * sizeof(hash_type),
                                          cudaMemcpyDeviceToHost,
                                          stream.get()));
    stream.sync();
    entries.resize(static_cast<std::size_t>(state.count));
    return entries;
  }

  [[nodiscard]] std::vector<std::uint8_t> serialize_compact(::cuda::stream_ref stream) const
  {
    device_state state{};
    std::vector<hash_type> entries(k_());
    DATASKETCHES_CUDA_TRY(
      cudaMemcpyAsync(&state, state_.data(), sizeof(state), cudaMemcpyDeviceToHost, stream.get()));
    DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(entries.data(),
                                          hashes_.data(),
                                          entries.size() * sizeof(hash_type),
                                          cudaMemcpyDeviceToHost,
                                          stream.get()));
    stream.sync();
    entries.resize(static_cast<std::size_t>(state.count));
    return serialize_compact_v3(
      state.empty != 0, get_seed_hash(), effective_theta_(state), entries);
  }

  void load_compact_(::cuda::stream_ref stream, const compact_image& image)
  {
    if (image.entries.size() > k_()) {
      throw std::invalid_argument(
        "theta_sketch::deserialize: retained entries exceed configured nominal k");
    }
    DATASKETCHES_CUDA_TRY(
      cudaMemsetAsync(hashes_.data(), 0xff, hashes_.size() * sizeof(hash_type), stream.get()));
    if (!image.entries.empty()) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(hashes_.data(),
                                            image.entries.data(),
                                            image.entries.size() * sizeof(hash_type),
                                            cudaMemcpyHostToDevice,
                                            stream.get()));
    }
    write_state_(stream,
                 device_state{image.empty ? starting_theta_(p_) : image.theta,
                              image.entries.size(),
                              image.empty ? 1U : 0U});
    stream.sync();
  }
};

}  // namespace datasketches::cuda::detail::theta
