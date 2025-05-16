# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import importlib.resources

# Read VERSION file from the module that is symlinked to VERSION file
# in the root of the repo at build time or copied to the module at
# installation. VERSION is a separate file that allows CI build-time scripts
# to update version info (including commit hashes) without modifying
# source files.
__version__ = (
    importlib.resources.files("_nx_hipgraph").joinpath("VERSION").read_text().strip()
)
__git_commit__ = ""
