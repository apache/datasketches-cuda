set(CPM_DOWNLOAD_VERSION 0.42.3)
file(DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
  ${CMAKE_BINARY_DIR}/CPM.cmake
  EXPECTED_HASH SHA256=a609e875fd532b067174250f6abbc3dac22fe2d64869783fb1e80bda1625c844)
include(${CMAKE_BINARY_DIR}/CPM.cmake)
