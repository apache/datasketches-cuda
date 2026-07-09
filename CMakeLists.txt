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

cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

project(datasketches_cuda VERSION 0.0.1 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT GENERATOR_IS_MULTI_CONFIG)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# BUILD_TESTS defaults ON when this is the top-level project and OFF when
# consumed via add_subdirectory or CPM (same convention as cuCollections).
set(_default OFF)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(_default ON)
endif()
option(BUILD_TESTS "Build unit tests" ${_default})

find_package(CUDAToolkit REQUIRED)

include(cmake/thirdparty/get_cccl.cmake)

# datasketches-cpp is a *public* dependency: include/datasketches/cuda/hll.hpp,
# detail/hll/preamble.hpp, detail/hll/composite_finalizer.hpp, and
# detail/hll/reduction_state.hpp all use upstream symbols (target_hll_type,
# HllUtil, CompositeInterpolationXTable, HarmonicNumbers, INVERSE_POWERS_OF_2).
# Configure it at top level so consumers via add_subdirectory/CPM inherit
# the include dirs and the `datasketches` target automatically.
include(cmake/thirdparty/get_datasketches_cpp.cmake)

# When the host C++ compiler is from a non-system prefix (e.g. a bundled GCC),
# embed its lib64 into the build RPATH so test binaries pick up the matching
# libstdc++.so without requiring LD_LIBRARY_PATH at run time.
get_filename_component(_dscuda_cxx_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
if(EXISTS "${_dscuda_cxx_dir}/../lib64/libstdc++.so.6")
  list(APPEND CMAKE_BUILD_RPATH "${_dscuda_cxx_dir}/../lib64")
endif()

add_library(datasketches_cuda INTERFACE)
add_library(datasketches::cuda ALIAS datasketches_cuda)
# Without an explicit EXPORT_NAME, install(EXPORT ... NAMESPACE datasketches::)
# would produce the IMPORTED target name `datasketches::datasketches_cuda` —
# different from the build-tree alias `datasketches::cuda`. Setting
# EXPORT_NAME=cuda gives consumers the same `datasketches::cuda` name in both
# add_subdirectory and find_package scenarios. The leaf `cuda` is always
# namespace-prefixed by CMake (`datasketches::cuda`); there is no consumer
# pathway where a bare `cuda` target can clash with anything.
set_target_properties(datasketches_cuda PROPERTIES EXPORT_NAME cuda)
target_include_directories(datasketches_cuda INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(datasketches_cuda INTERFACE CCCL::CCCL CUDA::cudart datasketches)
target_compile_features(datasketches_cuda INTERFACE cxx_std_17 cuda_std_17)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test/hll)
endif()

# Install rules — always present, same as datasketches-cpp's convention.
# Note: CCCL and datasketches are CPM-fetched and are not installed by this
# package. Downstream `find_package(datasketches_cuda CONFIG)` consumers must
# provide both on their CMAKE_PREFIX_PATH. Consumption via add_subdirectory
# or CPM works without any extra setup.
include(CMakePackageConfigHelpers)

install(DIRECTORY include/ DESTINATION include)

install(TARGETS datasketches_cuda EXPORT datasketches_cuda-targets)
install(EXPORT datasketches_cuda-targets
  NAMESPACE datasketches::
  DESTINATION lib/cmake/datasketches_cuda
  FILE datasketches_cuda-targets.cmake)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/datasketches_cuda-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/datasketches_cuda-config.cmake
  INSTALL_DESTINATION lib/cmake/datasketches_cuda)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/datasketches_cuda-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/datasketches_cuda-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/datasketches_cuda-config-version.cmake
  DESTINATION lib/cmake/datasketches_cuda)

# Developer-only format targets. Skipped when consumed via add_subdirectory/CPM.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  include(cmake/format.cmake)
endif()
