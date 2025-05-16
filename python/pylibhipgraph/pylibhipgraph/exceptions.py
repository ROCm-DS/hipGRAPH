# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""
Exception classes for pylibhipgraph.
"""


class FailedToConvergeError(Exception):
    """
    Raised when an algorithm fails to converge within a predetermined set of
    constraints which vary based on the algorithm, and may or may not be
    user-configurable.
    """

    pass
