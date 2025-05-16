# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Not included because of cugraph_ops. set(cython_sources random_walks_wrapper.pyx)
set(linked_libraries roc::hipgraph)
rapids_cython_create_modules(
    CXX
    SOURCE_FILES
    "${cython_sources}"
    LINKED_LIBRARIES
    "${linked_libraries}"
    MODULE_PREFIX
    sampling_
    ASSOCIATED_TARGETS
    hipgraph)
