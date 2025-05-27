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

if(BUILD_CODE_COVERAGE)
  add_custom_target(
    code_cov_tests
    DEPENDS hipgraph-test
    COMMAND ${CMAKE_COMMAND} -E rm -rf ./coverage-report
    COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="./coverage-report/profraw/hipgraph-coverage_%p.profraw" $<TARGET_FILE:hipgraph-test>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  find_program(
    LLVM_PROFDATA
    llvm-profdata
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  find_program(
    LLVM_COV
    llvm-cov
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  add_custom_target(
    coverage
    DEPENDS code_cov_tests
    COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/hipgraph-coverage_*.profraw -o ./coverage-report/hipgraph.profdata
    COMMAND ${LLVM_COV} report -object ./library/src/libhipgraph.so -instr-profile=./coverage-report/hipgraph.profdata
    COMMAND ${LLVM_COV} show -object ./library/src/libhipgraph.so -instr-profile=./coverage-report/hipgraph.profdata -format=html -output-dir=coverage-report
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

endif()


set_target_properties(hipgraph-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

rocm_install(TARGETS hipgraph-test COMPONENT tests)

add_test(hipgraph-test hipgraph-test)
