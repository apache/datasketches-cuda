# Adds `format` and `format-check` targets driven by clang-format. Skipped
# silently when consumed via add_subdirectory/CPM; warns at top-level configure
# if `clang-format` is not on PATH.
#
# Glob list is recomputed at re-configure (file(GLOB_RECURSE CONFIGURE_DEPENDS)),
# so newly added source files are picked up without manual edits to this file.

find_program(CLANG_FORMAT clang-format)
if(NOT CLANG_FORMAT)
  message(WARNING
    "datasketches_cuda: clang-format not found on PATH; `format` and "
    "`format-check` targets unavailable")
  return()
endif()

file(GLOB_RECURSE _FMT_FILES
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/../include/*.hpp
  ${CMAKE_CURRENT_LIST_DIR}/../include/*.cuh
  ${CMAKE_CURRENT_LIST_DIR}/../include/*.cu
  ${CMAKE_CURRENT_LIST_DIR}/../include/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../include/*.inl
  ${CMAKE_CURRENT_LIST_DIR}/../test/*.hpp
  ${CMAKE_CURRENT_LIST_DIR}/../test/*.cuh
  ${CMAKE_CURRENT_LIST_DIR}/../test/*.cu
  ${CMAKE_CURRENT_LIST_DIR}/../test/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../test/*.inl)

add_custom_target(format
  COMMAND ${CLANG_FORMAT} -i ${_FMT_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
  COMMENT "Running clang-format -i on source tree"
  VERBATIM)

add_custom_target(format-check
  COMMAND ${CLANG_FORMAT} --dry-run --Werror ${_FMT_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
  COMMENT "Checking clang-format compliance"
  VERBATIM)
