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

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time This
# tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print verbose compiler flags
if(BUILD_VERBOSE)
    include(../cmake/Verbose.cmake)
endif()

# Configure a header file to pass the hipGRAPH version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipgraph/hipgraph-version.h.in"
               "${PROJECT_BINARY_DIR}/include/hipgraph/hipgraph-version.h")

# Copy Header files to build directory
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipgraph/hipgraph.h"
               "${PROJECT_BINARY_DIR}/include/hipgraph/hipgraph.h" COPYONLY)

# Public hipGRAPH headers TODO: Stop maintaining these manually.
set(hipgraph_headers_public include/hipgraph/hipgraph.h ${PROJECT_BINARY_DIR}/include/hipgraph/hipgraph-version.h)
set(hipgraph_c_headers_public
    include/hipgraph/hipgraph_c/algorithms.h
    include/hipgraph/hipgraph_c/array.h
    include/hipgraph/hipgraph_c/centrality_algorithms.h
    include/hipgraph/hipgraph_c/community_algorithms.h
    include/hipgraph/hipgraph_c/core_algorithms.h
    include/hipgraph/hipgraph_c/error.h
    include/hipgraph/hipgraph_c/graph_functions.h
    include/hipgraph/hipgraph_c/graph_generators.h
    include/hipgraph/hipgraph_c/graph.h
    include/hipgraph/hipgraph_c/labeling_algorithms.h
    include/hipgraph/hipgraph_c/random.h
    include/hipgraph/hipgraph_c/resource_handle.h
    include/hipgraph/hipgraph_c/sampling_algorithms.h
    include/hipgraph/hipgraph_c/similarity_algorithms.h
    include/hipgraph/hipgraph_c/traversal_algorithms.h)

set(hipgraph_cpp_headers_not_quite_public
    include/hipgraph/cpp/algorithms.hpp
    include/hipgraph/cpp/api_helpers.hpp
    include/hipgraph/cpp/dendrogram.hpp
    include/hipgraph/cpp/edge_partition_device_view_device.hpp
    include/hipgraph/cpp/edge_partition_edge_property_device_view_device.hpp
    include/hipgraph/cpp/edge_partition_endpoint_property_device_view_device.hpp
    include/hipgraph/cpp/edge_partition_view.hpp
    include/hipgraph/cpp/edge_property.hpp
    include/hipgraph/cpp/edge_src_dst_property.hpp
    include/hipgraph/cpp/eidecl_graph.hpp
    include/hipgraph/cpp/graph.hpp
    include/hipgraph/cpp/graph_functions.hpp
    include/hipgraph/cpp/graph_generators.hpp
    include/hipgraph/cpp/graph_view.hpp
    include/hipgraph/cpp/legacy/eidecl_graph.hpp
    include/hipgraph/cpp/legacy/functions.hpp
    include/hipgraph/cpp/legacy/graph.hpp
    include/hipgraph/cpp/legacy/internals.hpp
    include/hipgraph/cpp/mtmg/edge_property.hpp
    include/hipgraph/cpp/mtmg/edge_property_view.hpp
    include/hipgraph/cpp/mtmg/edgelist.hpp
    include/hipgraph/cpp/mtmg/graph.hpp
    include/hipgraph/cpp/mtmg/graph_view.hpp
    include/hipgraph/cpp/mtmg/handle.hpp
    include/hipgraph/cpp/mtmg/instance_manager.hpp
    include/hipgraph/cpp/mtmg/per_thread_edgelist.hpp
    include/hipgraph/cpp/mtmg/renumber_map.hpp
    include/hipgraph/cpp/mtmg/renumber_map_view.hpp
    include/hipgraph/cpp/mtmg/resource_manager.hpp
    include/hipgraph/cpp/mtmg/vertex_pair_result.hpp
    include/hipgraph/cpp/mtmg/vertex_pair_result_view.hpp
    include/hipgraph/cpp/mtmg/vertex_result.hpp
    include/hipgraph/cpp/mtmg/vertex_result_view.hpp
    include/hipgraph/cpp/partition_manager.hpp
    include/hipgraph/cpp/sampling_functions.hpp
    include/hipgraph/cpp/utilities/atomic_ops_device.hpp
    include/hipgraph/cpp/utilities/dataframe_buffer.hpp
    include/hipgraph/cpp/utilities/device_comm.hpp
    include/hipgraph/cpp/utilities/device_functors_device.hpp
    include/hipgraph/cpp/utilities/device_properties.hpp
    include/hipgraph/cpp/utilities/error.hpp
    include/hipgraph/cpp/utilities/graph_traits.hpp
    include/hipgraph/cpp/utilities/host_scalar_comm.hpp
    include/hipgraph/cpp/utilities/mask_utils_device.hpp
    include/hipgraph/cpp/utilities/misc_utils_device.hpp
    include/hipgraph/cpp/utilities/packed_bool_utils.hpp
    include/hipgraph/cpp/utilities/path_retrieval.hpp
    include/hipgraph/cpp/utilities/shuffle_comm_device.hpp
    include/hipgraph/cpp/utilities/thrust_tuple_utils.hpp
    include/hipgraph/cpp/vertex_partition_device_view_device.hpp
    include/hipgraph/cpp/vertex_partition_view.hpp
    include/hipgraph/cpp/edge_partition_endpoint_property_device_view_device.hpp
    include/hipgraph/cpp/vertex_partition_device_view_device.hpp
    include/hipgraph/cpp/edge_partition_edge_property_device_view_device.hpp
    include/hipgraph/cpp/utilities/device_functors_device.hpp
    include/hipgraph/cpp/utilities/shuffle_comm_device.hpp
    include/hipgraph/cpp/utilities/atomic_ops_device.hpp
    include/hipgraph/cpp/utilities/misc_utils_device.hpp
    include/hipgraph/cpp/utilities/mask_utils_device.hpp
    include/hipgraph/cpp/edge_partition_device_view_device.hpp
    include/hipgraph/cpp/detail/decompress_edge_partition_device.hpp)

source_group(
    "Header Files\\Public"
    FILES ${hipgraph_headers_public}
          # TODO: Check if this matches the cugraph layout under Windows, if cugraph support exists.
          ${hipgraph_c_headers_public} ${hipgraph_cpp_headers_not_quite_public})

# Create hipGRAPH library
add_library(hipgraph ${hipgraph_headers_public} ${hipgraph_c_headers_public})
add_library(hip::hipgraph ALIAS hipgraph)
add_library(roc::hipgraph ALIAS hipgraph)

# Target compile options
target_compile_options(hipgraph PRIVATE -Wno-unused-command-line-argument -Wall)
# Target include directories
target_include_directories(
    hipgraph
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/hipgraph> # For use *in* library/src
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipgraph> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
           $<INSTALL_INTERFACE:include>)

# Target properties
rocm_set_soversion(hipgraph ${hipgraph_SOVERSION})
set_target_properties(hipgraph PROPERTIES CXX_EXTENSIONS NO)
set_target_properties(
    hipgraph
    PROPERTIES C_VISIBILITY_PRESET "hidden"
               CXX_VISIBILITY_PRESET "hidden"
               VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(hipgraph PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_properties(hipgraph PROPERTIES DEBUG_POSTFIX "-d")

# Following boost conventions of prefixing 'lib' on static built libraries
if(NOT BUILD_SHARED_LIBS)
    set_target_properties(hipgraph PROPERTIES PREFIX "lib")
endif()

if(BUILD_CODE_COVERAGE)
  target_compile_options( hipgraph PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping )
  target_link_options( hipgraph PUBLIC -fprofile-instr-generate )
endif()

# Include sources
add_subdirectory(src)

include(GenerateExportHeader)
generate_export_header(hipgraph EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipgraph/hipgraph-export.h)

# Install targets
rocm_install_targets(TARGETS hipgraph INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)

# DEPENDS are set in the *_detail directories.
rocm_export_targets(
    TARGETS roc::hipgraph
    DEPENDS PACKAGE hip "${HIPGRAPH_BACKEND_DEPENDS}"
    NAMESPACE roc::)
