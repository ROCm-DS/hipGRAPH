# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import importlib.resources

__version__ = (
    importlib.resources.files("hipgraph._pylibraft")
    .joinpath("VERSION")
    .read_text()
    .strip()
)
__git_commit__ = ""
