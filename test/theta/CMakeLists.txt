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

function(ConfigureThetaTest TEST_NAME)
  add_executable(${TEST_NAME} ${ARGN})
  target_link_libraries(${TEST_NAME} PRIVATE
    Catch2::Catch2WithMain
    datasketches::cuda
    datasketches
    CUDA::cudart)
  set_target_properties(${TEST_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  catch_discover_tests(${TEST_NAME} EXTRA_ARGS --allow-running-no-tests)
endfunction()

ConfigureThetaTest(THETA_BASIC_TEST basic_sketch_test.cu cpu_reference.cpp)
ConfigureThetaTest(THETA_SET_OPERATIONS_TEST set_operations_test.cu cpu_reference.cpp)
ConfigureThetaTest(THETA_TYPED_PARITY_TEST parity_typed_test.cu cpu_reference.cpp)
