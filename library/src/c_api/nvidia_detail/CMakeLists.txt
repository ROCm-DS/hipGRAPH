# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
cmake_policy(PUSH)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0082 NEW)
cmake_policy(SET CMP0076 NEW)

include(CMakePrintHelpers)

find_package(CUDAToolkit REQUIRED)

target_compile_definitions(hipgraph PUBLIC __HIP_PLATFORM_NVIDIA__)
target_include_directories(hipgraph PUBLIC $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
                                           $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>)
if(DEFINED ENV{CONDA_PREFIX} AND NOT CUGRAPH_BUILD_DIR)
    target_include_directories(hipgraph PUBLIC $<BUILD_INTERFACE:$ENV{CONDA_PREFIX}/include>)
endif()

# Find the cugraph C interface library. This could be from the toolkit in general, RAPIDS, or a build directory. RAPIDS
# uses conda. In that environment, the RAPIDS libraries are in ENV{CONDA_PREFIX}/lib at the moment but may move into the
# CUDA Toolkit someday.
#
# If the CUDA Toolkit is under ENV{CONDA_PREFIX}, then it'll be in a target-specific subdirectory. So
# ${CUDAToolkit_LIBRARY_DIR}/../../lib could point at the library as well.
#
# Once found, we currently assume the RAPIDS libraries that are not in the CUDAToolkit targets are in the same location.
#
# The next problem is if there is a CUGRAPH_BUILD_DIR. Then the required dependencies are under _deps/ within that build
# dir. But... Life is simpler if we assume a conda-based *installation* that contains the dependencies.

if(NOT CUGRAPH_BUILD_DIR)
    set(cugraph_hints ${CUDAToolkit_LIBRARY_DIR}/../../lib ${CUDAToolkit_LIBRARY_DIR})
    if(DEFINED ENV{CONDA_PREFIX})
        list(PREPEND cugraph_hints $ENV{CONDA_PREFIX}/lib)
    endif()
    find_library(
        CUDA_cugraph_LIBRARY
        NAMES cugraph
        HINTS ${cugraph_hints} REQUIRED)
    find_library(
        CUDA_cugraph_c_LIBRARY
        NAMES cugraph_c
        HINTS ${cugraph_hints} REQUIRED)
else()
    find_library(
        CUDA_cugraph_LIBRARY
        NAMES cugraph
        PATHS ${CUGRAPH_BUILD_DIR} REQUIRED)
    find_library(
        CUDA_cugraph_c_LIBRARY
        NAMES cugraph_c
        PATHS ${CUGRAPH_BUILD_DIR} REQUIRED)
    set(_cugraph_inc ${CUGRAPH_BUILD_DIR}/../cpp/include)
    if(DEFINED CUGRAPH_SOURCE_DIR)
        list(APPEND _cugraph_inc ${CUGRAPH_SOURCE_DIR}/cpp/include)
    endif()
    find_file(
        _cugraph_includes
        NAMES cugraph_c/array.h
        HINTS ${_cugraph_inc})
    target_include_directories(hipgraph PUBLIC $<BUILD_INTERFACE:${_cugraph_includes}>)
endif()

# cmake_print_variables(CUDA_cugraph_LIBRARY CUDA_cugraph_c_LIBRARY)

# From the toolkit, we use CUDA::cudart CUDA::cublas, CUDA::cublasLt. CUDA::curand, CUDA::cusolver, CUDA::cusparse
# targets for linking the clients.
#
# Some of these depend on others and hopefully pull in everything cugraph needs...
#
# TODO: Figure out the static linking varieties.
set(HIPGRAPH_BACKEND_DEPENDS ${CUDA_cugraph_c_LIBRARY} ${CUDA_cugraph_LIBRARY} CUDA::cusolver CUDA::curand
                             CUDA::cublasLt CUDA::cusparse CUDA::cublas CUDA::cudart)
target_link_libraries(hipgraph PRIVATE ${HIPGRAPH_BACKEND_DEPENDS})

# hipGRAPH CUDA source
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

cmake_policy(POP)
