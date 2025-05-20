# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import networkx as nx
import nx_hipgraph as nxcg
import pytest


@pytest.mark.parametrize(
    "get_graph", [nx.florentine_families_graph, nx.les_miserables_graph]
)
def test_k_truss(get_graph):
    Gnx = get_graph()
    Gcg = nxcg.from_networkx(Gnx, preserve_all_attrs=True)
    for k in range(6):
        Hnx = nx.k_truss(Gnx, k)
        Hcg = nxcg.k_truss(Gcg, k)
        assert nx.utils.graphs_equal(Hnx, nxcg.to_networkx(Hcg))
        if Hnx.number_of_edges() == 0:
            break
