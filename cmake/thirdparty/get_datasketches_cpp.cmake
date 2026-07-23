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

# apache/datasketches-cpp: header-only library used as:
#   - parity reference (CPU sketch) in tests
#   - public dependency for target_hll_type, HllUtil, wire-format constants,
#     and inverse-power tables used by installed headers
#
# Minimum version: 5.0.0. The required HLL types and helpers have been in the
# public surface since at least v3.x; 5.0.0 is a conservative floor aligned
# with the pinned CPM fallback release, 5.2.0.
#
# Developer override: -DCPM_datasketches_SOURCE=/path/to/local/checkout (CPM-native).
# Note: if find_package succeeds, the CPM_datasketches_SOURCE override is ignored
# because the CPM fallback path is never reached.
if(NOT COMMAND CPMAddPackage)
  include(${CMAKE_CURRENT_LIST_DIR}/../get_cpm.cmake)
endif()

function(find_and_configure_datasketches_cpp)
  # datasketches-cpp's project name is `DataSketches` (CapitalCase) per its
  # top-level `project(DataSketches ...)`, so its installed config is
  # DataSketchesConfig.cmake and the found-var is DataSketches_FOUND. The
  # in-source target name remains lowercase `datasketches` (unaliased), which
  # is what we link against. The CPM NAME below is also `datasketches` because
  # that's the in-source package name CPM expects to match against
  # CPM_datasketches_SOURCE overrides.
  find_package(DataSketches 5.0.0 CONFIG QUIET)
  if(DataSketches_FOUND)
    message(STATUS
      "datasketches_cuda: using installed DataSketches ${DataSketches_VERSION}")
    return()
  endif()

  message(STATUS
    "datasketches_cuda: DataSketches >= 5.0.0 not found via find_package "
    "- fetching 5.2.0 via CPM")
  CPMAddPackage(
    NAME datasketches
    GITHUB_REPOSITORY apache/datasketches-cpp
    GIT_TAG 5.2.0
    OPTIONS
      "BUILD_TESTS OFF"
  )
endfunction()

find_and_configure_datasketches_cpp()
