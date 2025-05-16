# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import networkx as nx


class Graph(nx.Graph):
    """
    Class which extends NetworkX Graph class. It provides original
    NetworkX functionality and will be overridden as this compatibility
    layer moves functionality to gpus in future releases.
    """

    pass
