#pragma once

#include <cmath>
#include <cstdint>

#include <CompositeInterpolationXTable.hpp>
#include <CubicInterpolation.hpp>
#include <HarmonicNumbers.hpp>

namespace datasketches::cuda::detail {

//! @brief Returns the HLL "raw" estimator (HyperLogLog harmonic mean form with
//! the small-k correction factor).
//!
//! Mirrors `HllArray::getHllRawEstimate` (`HllArray-internal.hpp:577-587`).
//!
//! @param[in] kxq0_plus_kxq1 The sum of `kxq0_` and `kxq1_` from the wider
//!   register-array reduction (which is `sum_i 2^{-r_i}` split for precision).
//! @param[in] lgK The HLL precision parameter.
//! @return The bias-uncorrected HLL estimate.
inline double hll_raw_estimate(double kxq0_plus_kxq1, uint8_t lgK) noexcept
{
  const uint32_t configK = 1u << lgK;
  double correctionFactor;
  if (lgK == 4) {
    correctionFactor = 0.673;
  } else if (lgK == 5) {
    correctionFactor = 0.697;
  } else if (lgK == 6) {
    correctionFactor = 0.709;
  } else {
    correctionFactor = 0.7213 / (1.0 + (1.079 / configK));
  }
  return (correctionFactor * configK * configK) / kxq0_plus_kxq1;
}

//! @brief Returns the linear-counting (bit-map) estimate used by Composite at
//! low cardinalities.
//!
//! Mirrors `HllArray::getHllBitMapEstimate` (`HllArray-internal.hpp:563-574`).
//! Delegates to `datasketches::HarmonicNumbers::getBitMapEstimate` for the
//! actual harmonic-number formula.
//!
//! @param[in] curMin The minimum register value (always 0 for HLL_8 with at
//!   least one zero register).
//! @param[in] numAtCurMin Count of registers equal to `curMin`. When
//!   `curMin == 0`, this is the count of zero (unhit) registers.
//! @param[in] lgK The HLL precision parameter.
//! @return The linear-counting estimate.
inline double hll_bitmap_estimate(uint8_t curMin, uint32_t numAtCurMin, uint8_t lgK)
{
  const uint32_t configK         = 1u << lgK;
  const uint32_t numUnhitBuckets = (curMin == 0) ? numAtCurMin : 0u;
  if (numUnhitBuckets == 0) { return configK * std::log(configK / 0.5); }
  const uint32_t numHitBuckets = configK - numUnhitBuckets;
  return ::datasketches::HarmonicNumbers<>::getBitMapEstimate(static_cast<int>(configK),
                                                              static_cast<int>(numHitBuckets));
}

//! @brief Composite ("non-HIP") cardinality estimator.
//!
//! Mirrors `HllArray::getCompositeEstimate` (`HllArray-internal.hpp:367-409`)
//! verbatim, but takes the wider reduction state as parameters instead of
//! reading them from an `HllArray` instance. The interpolation and harmonic
//! helpers are reached via nested-namespace lookup into `datasketches::`.
//!
//! @param[in] kxq0_plus_kxq1 `kxq0 + kxq1` from the wider reduction.
//! @param[in] curMin Minimum register value across the array.
//! @param[in] numAtCurMin Count of registers equal to `curMin`.
//! @param[in] lgK The HLL precision parameter (4..21).
//! @return The Composite cardinality estimate.
inline double composite_finalizer(double kxq0_plus_kxq1,
                                  uint8_t curMin,
                                  uint32_t numAtCurMin,
                                  uint8_t lgK)
{
  const double rawEst = hll_raw_estimate(kxq0_plus_kxq1, lgK);

  const double* xArr     = ::datasketches::CompositeInterpolationXTable<>::get_x_arr(lgK);
  const uint32_t xArrLen = ::datasketches::CompositeInterpolationXTable<>::get_x_arr_length();
  const double yStride   = ::datasketches::CompositeInterpolationXTable<>::get_y_stride(lgK);

  if (rawEst < xArr[0]) { return 0.0; }

  const uint32_t xArrLenM1 = xArrLen - 1;

  if (rawEst > xArr[xArrLenM1]) {
    const double finalY = yStride * xArrLenM1;
    const double factor = finalY / xArr[xArrLenM1];
    return rawEst * factor;
  }

  const double adjEst = ::datasketches::CubicInterpolation<>::usingXArrAndYStride(
    xArr, static_cast<int>(xArrLen), yStride, rawEst);

  // Empirical: avoid the linear-counting estimator if it might have a crazy
  // value. Threshold 3*k is safe for 2^4 <= k <= 2^21.
  if (adjEst > static_cast<double>(3u << lgK)) { return adjEst; }

  const double linEst = hll_bitmap_estimate(curMin, numAtCurMin, lgK);
  const double avgEst = (adjEst + linEst) / 2.0;

  // Empirical crossover constants (HllArray-internal.hpp:404-406).
  double crossOver = 0.64;
  if (lgK == 4) {
    crossOver = 0.718;
  } else if (lgK == 5) {
    crossOver = 0.672;
  }

  return (avgEst > (crossOver * static_cast<double>(1u << lgK))) ? adjEst : linEst;
}

}  // namespace datasketches::cuda::detail
