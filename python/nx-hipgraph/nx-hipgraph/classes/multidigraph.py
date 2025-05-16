# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from __future__ import annotations

import networkx as nx
import nx_hipgraph as nxcg

from .digraph import DiGraph
from .multigraph import MultiGraph

__all__ = ["MultiDiGraph"]

networkx_api = nxcg.utils.decorators.networkx_class(nx.MultiDiGraph)


class MultiDiGraph(MultiGraph, DiGraph):
    @classmethod
    @networkx_api
    def is_directed(cls) -> bool:
        return True

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiDiGraph]:
        return nx.MultiDiGraph

    ##########################
    # NetworkX graph methods #
    ##########################

    @networkx_api
    def to_undirected(self, reciprocal=False, as_view=False):
        raise NotImplementedError
