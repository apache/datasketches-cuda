#pragma once

#include <stdexcept>

namespace datasketches::cuda {

//! @brief Exception thrown when a CUDA runtime call fails.
class cuda_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace datasketches::cuda
