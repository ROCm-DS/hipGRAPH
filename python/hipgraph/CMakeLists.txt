# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Keeping this in the file to limit diff noise during debugging.
include(CMakePrintHelpers)

project(
    hipgraph-python
    VERSION 0.99.0
    LANGUAGES CXX HIP)

set(ROCM_DIRECTORY
    "/opt/rocm"
    CACHE PATH "Path to an installed ROCm" FORCE)
mark_as_advanced(ROCM_DIRECTORY)

find_package(hip REQUIRED CONFIG HINTS "$ENV{ROCM_PATH}" "${ROCM_DIRECTORY}" /opt/rocm /usr)
find_package(amd_comgr REQUIRED CONFIG HINTS "$ENV{ROCM_PATH}" "${ROCM_DIRECTORY}" /opt/rocm /usr)
include_directories(${HIP_INCLUDE_DIRS})
link_libraries(amd_comgr)

# Import ROCm hipMM
include(FetchContent)
set(ROCGRAPH_HIPMM_BRANCH release/2.0.x)
message(STATUS "Fetching hipMM @ ${ROCGRAPH_HIPMM_BRANCH}")
fetchcontent_declare(
    rmm
    GIT_REPOSITORY https://$ENV{GITHUB_USER}:$ENV{GITHUB_PASS}@github.com/ROCm-DS/hipMM
    GIT_TAG ${ROCGRAPH_HIPMM_BRANCH}
    UPDATE_COMMAND git stash --all
    PATCH_COMMAND rm CMakeLists.txt)
fetchcontent_makeavailable(rmm)
include_directories("${rmm_SOURCE_DIR}/include")

set(ROCGRAPH_LIBHIPCXX_BRANCH release/2.2.x)
message(STATUS "Fetching libhipcxx @ ${ROCGRAPH_LIBHIPCXX_BRANCH}")
fetchcontent_declare(
    libhipcxx
    GIT_REPOSITORY https://$ENV{GITHUB_USER}:$ENV{GITHUB_PASS}@github.com/ROCm/libhipcxx.git
    GIT_TAG ${ROCGRAPH_LIBHIPCXX_BRANCH}
    UPDATE_COMMAND git stash --all
    PATCH_COMMAND rm CMakeLists.txt)
fetchcontent_makeavailable(libhipcxx)
include_directories(rocgraph "${libhipcxx_SOURCE_DIR}/include")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/.early-access-unreleased/raft/cpp/include/")

# AMD: Fix hipGRAPH's CMake config to link rocGRAPH correctly.
find_package(rocgraph CONFIG REQUIRED HINTS "$ENV{HIPGRAPH_PATH}" "${HIPGRAPH_PATH}" "${ROCM_DIRECTORY}" /opt/rocm)
link_libraries(roc::rocgraph)
find_package(hipgraph CONFIG REQUIRED HINTS "$ENV{HIPGRAPH_PATH}" "${HIPGRAPH_PATH}" "${ROCM_DIRECTORY}" /opt/rocm)
include_directories(${HIPGRAPH_INCLUDE_DIRS})
link_libraries(roc::hipgraph)

find_package(
    Python3
    COMPONENTS Interpreter Development.Module
    REQUIRED)

fetchcontent_declare(fmt URL https://github.com/fmtlib/fmt/archive/refs/tags/11.1.3.tar.gz)
fetchcontent_makeavailable(fmt)

link_libraries(fmt::fmt-header-only)

fetchcontent_declare(spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.1.tar.gz)
fetchcontent_makeavailable(spdlog)

link_libraries(spdlog::spdlog_header_only)
add_definitions(-DSPDLOG_FMT_EXTERNAL_HO)

find_package(fmt)
find_package(spdlog)
link_libraries(fmt::fmt-header-only)
link_libraries(spdlog::spdlog_header_only)
add_definitions(-DSPDLOG_FMT_EXTERNAL_HO)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/hipgraph/_pylibraft/common)

include(cmake/rapids-cython.cmake)

rapids_cython_init()
string(APPEND CYTHON_FLAGS " -I${HIP_INCLUDE_DIRS}" " -I${rmm_SOURCE_DIR}/include")

add_definitions(-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

add_subdirectory(hipgraph/_pylibraft/common)
add_subdirectory(hipgraph/components)
add_subdirectory(hipgraph/internals)
add_subdirectory(hipgraph/layout)
add_subdirectory(hipgraph/linear_assignment)
add_subdirectory(hipgraph/sampling)
add_subdirectory(hipgraph/structure)
add_subdirectory(hipgraph/tree)
add_subdirectory(hipgraph/utilities)
