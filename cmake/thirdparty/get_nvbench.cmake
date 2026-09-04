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

# NVIDIA/nvbench: benchmark harness, used only by BUILD_BENCHMARKS targets and
# never by the installed interface library.
#
# nvbench brings its own CCCL. NVBench_ENABLE_CUPTI is left at its default and
# NVBench_ENABLE_NVML is disabled so the benchmarks build on machines without
# the NVML development package.
#
# Developer override: -DCPM_nvbench_SOURCE=/path/to/local/checkout (CPM-native).
if(NOT COMMAND CPMAddPackage)
  include(${CMAKE_CURRENT_LIST_DIR}/../get_cpm.cmake)
endif()

function(find_and_configure_nvbench)
  CPMAddPackage(
    NAME nvbench
    GITHUB_REPOSITORY NVIDIA/nvbench
    GIT_TAG main
    OPTIONS
      "NVBench_ENABLE_NVML OFF"
      "NVBench_ENABLE_EXAMPLES OFF"
      "NVBench_ENABLE_TESTING OFF"
  )
endfunction()

find_and_configure_nvbench()
