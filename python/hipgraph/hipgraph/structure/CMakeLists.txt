# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

set(cython_sources graph_primtypes_wrapper.pyx graph_primtypes.pyx utils_wrapper.pyx)
# set(linked_libraries roc::hipgraph)

rapids_cython_create_modules(
    CXX
    SOURCE_FILES
    "${cython_sources}"
    LINKED_LIBRARIES
    "${linked_libraries}"
    MODULE_PREFIX
    structure_
    ASSOCIATED_TARGETS
    hipgraph)

# foreach(tgt IN LISTS "${RAPIDS_CYTHON_CREATED_TARGETS}") target_link_libraries("${tgt}" fmt::fmt-header-only
# spdlog::spdlog_header_only) target_compile_definitions("${tgt}" FMT_HEADER_ONLY SPDLOG_FMT_EXTERNAL_HO)
# target_include_directories("${tgt}" "${CMAKE_CURRENT_LIST_DIR}") endforeach()
