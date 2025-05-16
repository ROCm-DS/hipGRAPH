# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import hipgraph


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(hipgraph.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(hipgraph.__version__, str)
    assert len(hipgraph.__version__) > 0
