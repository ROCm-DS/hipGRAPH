# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# include(../../rapids_config.cmake)

# Keeping this in the file to limit diff noise during debugging.
include(CMakePrintHelpers)

project(
    pylibhipgraph
    VERSION 0.99.0
    LANGUAGES CXX)

set(ROCM_DIRECTORY
    "/opt/rocm"
    CACHE PATH "Path to an installed ROCm" FORCE)
mark_as_advanced(ROCM_DIRECTORY)

find_package(hip REQUIRED CONFIG HINTS "${ROCM_DIRECTORY}" /opt/rocm /usr)
set(CMAKE_C_COMPILER "${HIP_HIPCC_EXECUTABLE}")
set(CMAKE_CXX_COMPILER "${HIP_HIPCC_EXECUTABLE}")

find_package(
    Python3
    COMPONENTS Interpreter Development.Module
    REQUIRED)

include(cmake/rapids-cython.cmake)

find_package(hipgraph CONFIG REQUIRED HINTS "${HIPGRAPH_PATH}" "${ROCM_DIRECTORY}" /opt/rocm)

rapids_cython_init()

add_subdirectory(pylibhipgraph)
