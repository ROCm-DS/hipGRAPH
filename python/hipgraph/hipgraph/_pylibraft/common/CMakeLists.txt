# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

set(cython_sources cuda.pyx handle.pyx interruptible.pyx)

# Build all of the Cython targets
rapids_cython_create_modules(CXX SOURCE_FILES "${cython_sources}" MODULE_PREFIX common_)
