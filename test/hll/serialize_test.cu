// GPU-side serialize: produce bytes, hand them to CPU `datasketches::hll_sketch`,
// confirm CPU's get_estimate matches GPU's get_estimate.

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <thrust/device_vector.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <hll.hpp>

#include <datasketches/cuda/hll.hpp>

TEST_CASE("GPU serialize bytes accepted by CPU deserialize", "[serialize]")
{
  using Catch::Approx;
  for (uint8_t lgK : {uint8_t{8}, uint8_t{12}, uint8_t{16}}) {
    const uint64_t n = (uint64_t{1} << lgK) * 64;  // saturate well past LIST/SET threshold

    std::vector<uint64_t> host_keys(n);
    std::mt19937_64 rng(0xFACEB00CULL ^ lgK);
    for (auto& k : host_keys)
      k = rng();
    thrust::device_vector<uint64_t> dev_keys = host_keys;

    datasketches::cuda::hll_sketch<uint64_t> gpu(lgK);
    gpu.update(dev_keys.begin(), dev_keys.end());

    auto bytes = gpu.serialize_compact();
    REQUIRE(bytes.size() == 40u + (1u << lgK));

    auto cpu = ::datasketches::hll_sketch::deserialize(bytes.data(), bytes.size());
    INFO("lgK=" << int(lgK) << " n=" << n);
    REQUIRE(cpu.get_estimate() == Approx(gpu.get_estimate()).epsilon(1e-12));
  }
}

TEST_CASE("GPU serialize -> GPU deserialize round-trip", "[serialize]")
{
  using Catch::Approx;
  const uint8_t lgK = 12;
  const uint64_t n  = 200'000;
  std::vector<uint64_t> host_keys(n);
  std::mt19937_64 rng(0xDEFEC8EDULL);
  for (auto& k : host_keys)
    k = rng();
  thrust::device_vector<uint64_t> dev_keys = host_keys;

  datasketches::cuda::hll_sketch<uint64_t> a(lgK);
  a.update(dev_keys.begin(), dev_keys.end());

  auto bytes = a.serialize_compact();
  auto b     = datasketches::cuda::hll_sketch<uint64_t>::deserialize(
    ::cuda::std::span<const std::uint8_t>{bytes.data(), bytes.size()});

  REQUIRE(a.get_estimate() == Approx(b.get_estimate()));
  REQUIRE(a.serialize_compact() == b.serialize_compact());  // byte-equal
}
