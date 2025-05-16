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

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# CMake policy defaults

# Support relative paths in target_sources, which then are translated to global paths.
cmake_policy(SET CMP0076 NEW)
# Permit target_link_libraries in subdirectories to affect the parent. Assists in separating *_detail dependencies.
cmake_policy(SET CMP0079 NEW)
# Run add_subdirectory *while* processing the current directory rather than postponing them until the end. This lets
# subdirectories declare their dependencies that may be used across other subdirectories.
cmake_policy(SET CMP0082 NEW)

include(CMakeDependentOption)

# Keeping this in the file to limit diff noise during debugging.
include(CMakePrintHelpers)

if(NOT python)
    set(python "python3") # default for linux
endif()

# The core ROCm installation prefix is either into /opt/rocm on non-Windows, or ${PROJECT_BINARY_DIR}/package on
# Windows. However, packaging for different Linux distributions or other scenarios may require different settings, so we
# do not set any here.
#
# TODO: The CPack packaging likely needs the installation prefix set.

# Temporary. Need to work out lower-level dependencies, a FindROCm, etc.
set(ROCM_PATH
    "/opt/rocm"
    CACHE PATH "Path to an installed ROCm" FORCE)
mark_as_advanced(ROCM_PATH)

# Pick up static and dynamic shared object files
#
# TODO: Manage this via dependencies and other settings rather than assuming locations and precedence.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip)

# If running under conda, assume everything will be installed under conda, and prefer the conda-installed includes and
# libraries.
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

set(_hipgraph_module_path ${CMAKE_PREFIX_PATH})
list(TRANSFORM _hipgraph_module_path APPEND "/cmake")

# CMake modules
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${_hipgraph_module_path})

# TODO: Remove debugging bits. cmake_print_variables(CMAKE_PREFIX_PATH CMAKE_MODULE_PATH)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and up
if(POLICY CMP0066)
    cmake_policy(SET CMP0066 NEW)
endif()

# hipGRAPH project
#
# Note: At the hipGRAPH level, only client languages are considered. HIP and CUDA may be enabled in the detail
# subdirectories, but ultimately shouldn't be necessary for the hipGRAPH bindings.
project(
    hipgraph
    LANGUAGES C CXX
    VERSION 0.99.0)

# Build flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build hipGRAPH as a shared library" ON)
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" OFF)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_DOCS "Build documentation" OFF)
option(USE_CUDA "Build hipGRAPH using CUDA backend" OFF)
# If CUGRAPH_BUILD_DIR is not set, look at CUGRAPH_SOURCE_DIR/cpp/build.
cmake_dependent_option(CUGRAPH_SOURCE_DIR "Point at a cugraph source directory" "" USE_CUDA "")
cmake_dependent_option(CUGRAPH_BUILD_DIR "Point at a cugraph build directory (often cpp/build)" "" USE_CUDA "")

# TODO: Push both this and the sanitizer clauses into included files, or, better, use CMake modules that can make the
# options available when the platform supports them.
if(BUILD_CODE_COVERAGE)
    add_compile_options(-fprofile-arcs -ftest-coverage)
    add_link_options(--coverage)
endif()

if(BUILD_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
    add_link_options(-fuse-ld=lld)
endif()

# Global dependencies
include(cmake/Dependencies.cmake)

# Setup version
rocm_setup_version(VERSION 0.99.0)
set(hipgraph_SOVERSION 0.99.0)

# hipGRAPH library
add_subdirectory(library)
# hipGRAPH tests and samples
if(BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_SAMPLES)
    add_subdirectory(clients)
endif()

#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(BUILD_CODE_COVERAGE)
    #
    # > make coverage_cleanup (clean coverage related files.) > make coverage GTEST_FILTER=<> will run: > make
    # coverage_analysis GTEST_FILTER=<> (analyze tests) > make coverage_output (generate html documentation)
    #

    #
    # Run coverage analysis
    #
    add_custom_target(
        coverage_analysis
        COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
        COMMAND ./clients/staging/hipgraph-test --gtest_filter=\"\${GTEST_FILTER}\"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    add_dependencies(coverage_analysis hipgraph)

    #
    # Prepare coverage output This little script is generated because the option '--gcov-tool <program name>' of lcov
    # cannot take arguments.
    #
    add_custom_target(
        coverage_output
        DEPENDS coverage_analysis
        COMMAND mkdir -p lcoverage
        COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
        COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
        COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
        COMMAND chmod +x llvm-gcov.sh)

    #
    # Generate coverage output.
    #
    add_custom_command(
        TARGET coverage_output
        COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o
                lcoverage/raw_main_coverage.info
        COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
        COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage)

    add_custom_target(coverage DEPENDS coverage_output)

    #
    # Coverage cleanup
    #
    add_custom_target(
        coverage_cleanup
        COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

# Package specific CPACK vars
if(NOT USE_CUDA)
    rocm_package_add_dependencies(DEPENDS "rocgraph >= 0.99.0")
endif()
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT AND Apache-2.0")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")

if(NOT USE_CUDA)
    set(package_name hipgraph)
else()
    set(package_name hipgraph-alt)
endif()

set(HIPGRAPH_CONFIG_DIR
    "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "Path placed into ldconfig file")

if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(tests)
endif()

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm GRAPH library"
    MAINTAINER "hipGRAPH Maintainer <hipgraph-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPGRAPH_CONFIG_DIR})

# Build docs
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()
