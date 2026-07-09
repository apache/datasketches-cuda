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

# NVIDIA/cccl provides CCCL (which includes cudax). This project is still under
# active development against unreleased cudax HyperLogLog APIs, so use CCCL main
# for now instead of a released tag.
#
# TODO(find_package): once NVIDIA/cccl ships a tagged release containing the
# required cudax HyperLogLog policy and explicit stream / memory-resource APIs,
# replace the CPMAddPackage call below with the find_package-first pattern, e.g.:
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
if(NOT COMMAND CPMAddPackage)
  include(${CMAKE_CURRENT_LIST_DIR}/../get_cpm.cmake)
endif()

function(find_and_configure_cccl)
  message(WARNING
    "datasketches_cuda: fetching CCCL@main via CPM "
    "(TODO switch to a released CCCL version once required cudax HLL APIs are tagged)")
  CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY NVIDIA/cccl
    GIT_TAG main
    GIT_SHALLOW FALSE
    OPTIONS
      "CCCL_ENABLE_TESTING OFF"
      "CCCL_ENABLE_EXAMPLES OFF"
      "CCCL_ENABLE_BENCHMARKS OFF"
      "CCCL_ENABLE_UNSTABLE ON"
  )
endfunction()

find_and_configure_cccl()
