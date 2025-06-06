# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# ######################################################################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ######################################################################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

# Consider removing this in the future This should appear before the project command, because it does not use FORCE

set(CMAKE_INSTALL_PREFIX
    "/opt/rocm"
    CACHE PATH "Install path prefix, prepended onto install directories")

# This has to be initialized before the project() command appears Set the default of CMAKE_BUILD_TYPE to be release,
# unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

# This project may compile dependencies for clients.
project(hipgraph-clients LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time This
# tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT TARGET hipgraph)
    find_package(hipgraph CONFIG REQUIRED HINTS "${ROCM_DIRECTORY}" "${CMAKE_INSTALL_PREFIX}" "/opt/rocm")
    option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
    option(BUILD_CLIENTS_SAMPLES "Build examples" OFF)
endif()

# Build flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(BUILD_CLIENTS_SAMPLES)
    add_subdirectory(samples)
endif()

if(BUILD_CLIENTS_TESTS)
    enable_testing()
    add_subdirectory(tests)

    # The tests *currently* do neither download nor use any external matrices at the moment, so disabling these clauses.

    # if(NOT CONVERT_SOURCE) set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/deps/convert.cpp) endif() execute_process(COMMAND
    # ${CMAKE_CXX_COMPILER} ${CONVERT_SOURCE} -O3 -o ${PROJECT_BINARY_DIR}/mtx2csr.exe RESULT_VARIABLE STATUS) if(STATUS
    # AND NOT STATUS EQUAL 0) message(FATAL_ERROR "mtx2csr.exe failed to build, aborting.") endif()

    # set(HIPGRAPH_CLIENTMATRICES "${CMAKE_SOURCE_DIR}/cmake/hipgraph_clientmatrices.cmake") set(HIPGRAPH_CONVERT
    # "${PROJECT_BINARY_DIR}/hipgraph_mtx2csr")

    # add_custom_command( OUTPUT "${HIPGRAPH_CONVERT}" COMMAND ${CMAKE_COMMAND} -E copy
    # "${PROJECT_BINARY_DIR}/mtx2csr.exe" "${HIPGRAPH_CONVERT}" DEPENDS "${PROJECT_BINARY_DIR}/mtx2csr.exe"
    # WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    # add_custom_target(hipgraph-common DEPENDS "${HIPGRAPH_CLIENTMATRICES}" "${HIPGRAPH_CONVERT}")

    # rocm_install(PROGRAMS "${HIPGRAPH_CONVERT}" COMPONENT tests DESTINATION ${CMAKE_INSTALL_BINDIR})

    # rocm_install(FILES "${HIPGRAPH_CLIENTMATRICES}" COMPONENT tests DESTINATION
    # "${CMAKE_INSTALL_DATADIR}/hipgraph/test")

endif()
