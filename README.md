<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->

[![Maven Central](https://img.shields.io/maven-central/v/org.apache.datasketches/datasketches-cuda)](https://central.sonatype.com/artifact/org.apache.datasketches/datasketches-cuda)

# Apache® DataSketches™ Core CUDA Library Component
This is the core CUDA component of the DataSketches library.  It contains sketching algorithms that can be accessed directly from user applications.

Note that we have parallel core library components for Java, C++, Python, GO, and Rush implementations of many of the same sketch algorithms:

- [datasketches-java](https://github.com/apache/datasketches-java)
- [datasketches-cpp](https://github.com/apache/datasketches-cpp)
- [datasketches-python](https://github.com/apache/datasketches-python)
- [datasketches-go](https://github.com/apache/datasketches-go)
- [datasketches-rust](https://github.com/apache/datasketches-rust)

Please visit the main [DataSketches website](https://datasketches.apache.org) for more information.

If you are interested in making contributions to this site, please see our [Community](https://datasketches.apache.org/docs/Community/) page for how to contact us.

## Scope

This is a header-only INTERFACE library. The current release implements
HyperLogLog with the `HLL_8` target type, byte-compatible with
`datasketches::hll_sketch` for round-trip serialization. Other sketch families
and HLL variants are on the roadmap (see [Known Issues](#known-issues)).

Public header:

```cpp
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <datasketches/cuda/hll.hpp>

cuda::stream stream{cuda::devices[0]};
auto mr = cuda::device_default_memory_pool(cuda::devices[0]);

datasketches::cuda::hll_sketch<std::uint64_t> sketch(stream, mr, /*lgK=*/12);
sketch.update(stream, dev_keys.begin(), dev_keys.end());
double estimate = sketch.get_estimate(stream);

auto bytes = sketch.serialize_compact(stream);    // GPU -> CPU wire format
auto cpu   = datasketches::hll_sketch::deserialize(bytes.data(), bytes.size());
```

`hll_sketch` is a thin handle around `detail::hll::sketch_impl`, which in turn
owns a `cuda::experimental::cuco::hyperloglog` parameterized by a
`detail::hll::policy` (matching hash, bit-slicing, and seed). Construction and
CUDA-touching member functions take an explicit `cuda::stream_ref` as the first
argument; construction and deserialization also require an explicit device
memory resource. Streams used with `update_async` or `merge_async` must be
synchronized or otherwise ordered before the sketch is destroyed.

## Build & Runtime Dependencies

Required:

- CMake >= 3.30
- A C++17-capable host compiler (GCC 13.2+ verified; older GCC may work if it accepts C++17 and is supported by the CUDA toolkit)
- CUDA Toolkit >= 12.0 (12.4 verified)
- An NVIDIA GPU with compute capability supported by the active CUDA Toolkit (configured via `CMAKE_CUDA_ARCHITECTURES`; defaults to `native`)

Fetched automatically via CPM at configure time (no manual install required):

- [NVIDIA/cccl](https://github.com/NVIDIA/cccl) — pinned to commit `c95f99757cf95044ce82b905eec88ff40c851f7b` as synthetic version `3.5.1` while this library develops against unreleased cudax HLL APIs. This should move to a real CCCL release once the required APIs are tagged.
- [apache/datasketches-cpp](https://github.com/apache/datasketches-cpp) `5.2.0` (fall-back if `find_package(DataSketches 5.0.0 CONFIG)` does not locate a system install)
- [Catch2](https://github.com/catchorg/Catch2) `3.5.3` (test-only)

## Compilation and Test

Standard CMake workflow:

```bash
cmake -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Common options:

- `-DBUILD_TESTS=OFF` to skip building tests (defaults to `ON` at top level, `OFF` when consumed via `add_subdirectory`/CPM)
- `-DCMAKE_CUDA_ARCHITECTURES=<arch>` to target a specific GPU (e.g. `80` for A100; defaults to `native`)
- `-DCPM_CCCL_SOURCE=/path/to/local/cccl` to point CPM at a local CCCL checkout instead of fetching

Optional developer targets (added at top-level configure when `clang-format` is on `PATH`):

```bash
cmake --build build --target format        # format the tree in place
cmake --build build --target format-check  # dry-run, non-zero on diff
```

A `.pre-commit-config.yaml` is also provided for automatic formatting of
staged files on `git commit`. Install once with `pre-commit install`.

## Consuming the library

Either `add_subdirectory` / CPM:

```cmake
add_subdirectory(path/to/datasketches-cuda)
target_link_libraries(my_target PRIVATE datasketches::cuda)
```

Or `find_package` after installing:

```cmake
find_package(datasketches_cuda CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE datasketches::cuda)
```

Note: an installed `datasketches_cuda` does not propagate CCCL or
`datasketches-cpp` to consumers (both are CPM-fetched into the build tree).
Downstream `find_package` consumers must provide both on their
`CMAKE_PREFIX_PATH`. Consumption via `add_subdirectory` or CPM works without
any extra setup.

## Known Issues

- **HLL_8 only.** `HLL_4` and `HLL_6` packing are not yet implemented; constructing with those throws `std::invalid_argument`. `AuxHashMap` (the HLL_4 exception table) is also pending.
- **No LIST / SET deserialization.** The wire format's small-cardinality modes are rejected at parse. Sketches must already be in HLL mode.
- **Round-trip diverges on `FLAGS` (oooFlag) and `hipAccum`.** GPU output always sets `oooFlag=1` (pins CPU side to the Composite estimator) and `hipAccum=0` (no HIP tracking on parallel atomic update). All other bytes round-trip exactly.
- **CCCL uses a synthetic development version.** Until upstream tags a CCCL release containing the required cudax HLL policy and explicit stream / memory-resource APIs, `cmake/thirdparty/get_cccl.cmake` uses `CPMFindPackage` with synthetic version `3.5.1` and a pinned CCCL main commit. This prevents automatically accepting older CCCL installs from disk while keeping an explicit `CPM_CCCL_SOURCE` override available for development.
- **No driver on some dev hosts.** CI gates the runtime parity test (`parity_test.cu`); host-only tests (preamble, reduction state, normalizing hasher, composite finalizer, policy compile) pass without a GPU.
