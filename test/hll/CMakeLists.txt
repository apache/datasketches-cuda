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

CPMAddPackage(
  NAME Catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 3.5.3
)

if(Catch2_ADDED)
  list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
  include(Catch)
endif()

# Test targets get datasketches::cuda (our INTERFACE lib, brings CCCL + CUDA),
# datasketches (apache/datasketches-cpp INTERFACE lib for parity & helpers),
# and Catch2WithMain. CUDA::cudart is added explicitly for runtime calls in
# .cu tests; harmless for .cpp tests (unused link).
function(ConfigureTest TEST_NAME)
  add_executable(${TEST_NAME} ${ARGN})
  target_link_libraries(${TEST_NAME} PRIVATE
    Catch2::Catch2WithMain
    datasketches::cuda
    datasketches
    CUDA::cudart)
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(${TEST_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  catch_discover_tests(${TEST_NAME} EXTRA_ARGS --allow-running-no-tests)
endfunction()

ConfigureTest(NORMALIZING_HASHER_TEST normalizing_hasher_test.cpp)
ConfigureTest(POLICY_COMPILE_TEST policy_compile_test.cu)
ConfigureTest(COMPOSITE_FINALIZER_TEST composite_finalizer_test.cpp)
ConfigureTest(COMPOSITE_FINALIZER_DEVICE_TEST composite_finalizer_device_test.cu)
ConfigureTest(REDUCTION_STATE_TEST reduction_state_test.cpp)
ConfigureTest(PREAMBLE_TEST preamble_test.cpp)
ConfigureTest(BASIC_SKETCH_TEST basic_sketch_test.cu)
ConfigureTest(UNION_TEST union_test.cu)
ConfigureTest(SERIALIZE_TEST serialize_test.cu)
ConfigureTest(ROUND_TRIP_TEST round_trip_test.cu)
ConfigureTest(PARITY_TEST parity_test.cu)
ConfigureTest(PARITY_TYPED_TEST parity_typed_test.cu)
ConfigureTest(HLL_REF_TEST ref_compile_test.cu ref_test.cu)
