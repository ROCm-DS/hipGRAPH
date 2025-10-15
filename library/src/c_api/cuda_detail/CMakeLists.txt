# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
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

enable_language(HIP)
find_package(hip REQUIRED)

if(CUSTOM_ROCGRAPH)
    set(ENV{rocgraph_DIR} ${CUSTOM_ROCGRAPH})
    find_package(rocgraph REQUIRED CONFIG PATHS NO_CMAKE_PATH)
else()
    find_package(rocgraph REQUIRED CONFIG PATHS ${ROCGRAPH_PATH})
endif()

if(rocgraph_FOUND)
    message(STATUS "rocgraph package found")
else()
    message(FATAL_ERROR "rocgraph package not found")
endif()

target_compile_definitions(hipgraph PUBLIC __HIP_PLATFORM_AMD__)
target_include_directories(hipgraph PUBLIC $<BUILD_INTERFACE:${ROCGRAPH_INCLUDE_DIRS}/rocgraph>
                                           $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>)

set(hipgraph_source
    array.c
    centrality_algorithms.c
    community_algorithms.c
    core_algorithms.c
    graph_functions.c
    labeling_algorithms.c
    error.c
    graph.c
    graph_generators.c
    random.c
    resource_handle.c
    sampling_algorithms.c
    similarity_algorithms.c
    traversal_algorithms.c)
target_sources(hipgraph PRIVATE ${hipgraph_source})

target_link_libraries(hipgraph PUBLIC hip::host)
target_link_libraries(hipgraph PRIVATE roc::rocgraph)

set(HIPGRAPH_BACKEND_DEPENDS hip roc::rocgraph)
