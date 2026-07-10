# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# NVIDIA/cccl provides CCCL (which includes cudax). This project currently needs
# unreleased cudax HyperLogLog APIs. Pin a known-good CCCL main commit and assign
# it a synthetic version newer than the latest real CCCL release, so
# CPMFindPackage will not silently accept an older CCCL install from disk.
#
# TODO(find_package): once NVIDIA/cccl ships a tagged release containing the
# required cudax HyperLogLog policy and explicit stream / memory-resource APIs,
# replace the synthetic version and commit pin with that release version, e.g.:
#
#   find_package(CCCL X.Y.Z CONFIG QUIET COMPONENTS cudax)
#   if(CCCL_FOUND)
#     message(STATUS "datasketches_cuda: using installed CCCL ${CCCL_VERSION}")
#     return()
#   endif()
#   message(STATUS "datasketches_cuda: CCCL >= X.Y.Z not found -- fetching via CPM")
#
# CCCL_ENABLE_UNSTABLE must stay ON: cccl/CMakeLists.txt gates CCCL_ENABLE_CUDAX
# behind it, and hll/include/hll_sketch.hpp uses cuda::experimental::cuco::hyperloglog.
#
# Developer override: -DCPM_CCCL_SOURCE=/path/to/local/cccl (CPM-native).
if(NOT COMMAND CPMFindPackage)
  include(${CMAKE_CURRENT_LIST_DIR}/../get_cpm.cmake)
endif()

function(find_and_configure_cccl)
  set(_cccl_version 3.5.1)
  set(_cccl_tag c95f99757cf95044ce82b905eec88ff40c851f7b)
  message(WARNING
    "datasketches_cuda: using CCCL@${_cccl_tag} as synthetic version ${_cccl_version} "
    "(TODO switch to a released CCCL version once required cudax HLL APIs are tagged)")
  CPMFindPackage(
    NAME CCCL
    VERSION ${_cccl_version}
    GITHUB_REPOSITORY NVIDIA/cccl
    GIT_TAG ${_cccl_tag}
    GIT_SHALLOW FALSE
    FIND_PACKAGE_ARGUMENTS EXACT CONFIG COMPONENTS cudax
    OPTIONS
      "CCCL_ENABLE_TESTING OFF"
      "CCCL_ENABLE_EXAMPLES OFF"
      "CCCL_ENABLE_BENCHMARKS OFF"
      "CCCL_ENABLE_UNSTABLE ON"
  )
endfunction()

find_and_configure_cccl()
