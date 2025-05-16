# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# ######################################################################################################################
# Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

# Don't extract the timestamps.
cmake_policy(SET CMP0135 NEW)

set(old_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)

include(FetchContent)
set(GTEST_DEPENDENCIES)
fetchcontent_declare(googletest
                     URL https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz)
fetchcontent_makeavailable(googletest)

set(BUILD_SHARED_LIBS ${old_BUILD_SHARED_LIBS})

# Pre-declare the target.
add_executable(hipgraph-test hipgraph_gtest_main.cpp ../common/utility.cpp)
set_target_properties(hipgraph-test PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(c_api)

# Set GOOGLE_TEST definition
target_compile_definitions(hipgraph-test PRIVATE GOOGLE_TEST)

# Target compile options
# TODO: Fix the no-xyz warnings rather than ignoring them.
target_compile_options(hipgraph-test PRIVATE -Wall -Wno-unused-command-line-argument -Wno-vla-cxx-extension)

# Internal common header
target_include_directories(hipgraph-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# Target link libraries
target_link_libraries(hipgraph-test PRIVATE GTest::gtest_main roc::hipgraph)

if(USE_CUDA)
    target_compile_definitions(hipgraph-test PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(hipgraph-test PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(hipgraph-test PRIVATE ${CUDA_LIBRARIES})
endif()

set_target_properties(hipgraph-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

rocm_install(TARGETS hipgraph-test COMPONENT tests)

add_test(hipgraph-test hipgraph-test)
