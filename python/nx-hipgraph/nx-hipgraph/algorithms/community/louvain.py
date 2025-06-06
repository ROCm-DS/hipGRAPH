# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import warnings

import networkx as nx
import pylibhipgraph as plc
from nx_hipgraph.convert import _to_undirected_graph
from nx_hipgraph.utils import (
    _dtype_param,
    _groupby,
    _seed_to_int,
    networkx_algorithm,
    not_implemented_for,
)

__all__ = ["louvain_communities"]

# max_level argument was added to NetworkX 3.3
if nx.__version__[:3] <= "3.2":
    _max_level_param = {
        "max_level : int, optional": (
            "Upper limit of the number of macro-iterations (max: 500)."
        )
    }
else:
    _max_level_param = {}


def _louvain_communities_nx32(
    G,
    weight="weight",
    resolution=1,
    threshold=0.0000001,
    seed=None,
    *,
    max_level=None,
    dtype=None,
):
    """`seed` parameter is currently ignored, and self-loops are not yet supported."""
    return _louvain_communities(
        G, weight, resolution, threshold, max_level, seed, dtype=dtype
    )


def _louvain_communities(
    G,
    weight="weight",
    resolution=1,
    threshold=0.0000001,
    max_level=None,
    seed=None,
    *,
    dtype=None,
):
    """`seed` parameter is currently ignored, and self-loops are not yet supported."""
    # NetworkX allows both directed and undirected, but hipgraph only allows undirected.
    seed = _seed_to_int(seed)  # Unused, but ensure it's valid for future compatibility
    G = _to_undirected_graph(G, weight)
    if G.src_indices.size == 0:
        return [{key} for key in G._nodeiter_to_iter(range(len(G)))]
    if max_level is None:
        max_level = 500
    elif max_level > 500:
        warnings.warn(
            f"max_level is set too high (={max_level}), setting it to 500.",
            UserWarning,
            stacklevel=2,
        )
        max_level = 500
    node_ids, clusters, modularity = plc.louvain(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(weight, 1, dtype),
        max_level=max_level,  # TODO: add this parameter to NetworkX
        threshold=threshold,
        resolution=resolution,
        do_expensive_check=False,
    )
    groups = _groupby(clusters, node_ids, groups_are_canonical=True)
    return [set(G._nodearray_to_list(ids)) for ids in groups.values()]


_louvain_decorator = networkx_algorithm(
    extra_params={
        **_max_level_param,
        **_dtype_param,
    },
    is_incomplete=True,  # seed not supported; self-loops not supported
    is_different=True,  # RNG different
    version_added="23.10",
    _plc="louvain",
    name="louvain_communities",
)

if _max_level_param:  # networkx <= 3.2
    _louvain_communities_nx32.__name__ = "louvain_communities"
    louvain_communities = not_implemented_for("directed")(
        _louvain_decorator(_louvain_communities_nx32)
    )

    @louvain_communities._can_run
    def _(
        G,
        weight="weight",
        resolution=1,
        threshold=0.0000001,
        seed=None,
        *,
        max_level=None,
        dtype=None,
    ):
        # NetworkX allows both directed and undirected, but hipgraph only undirected.
        return not G.is_directed()

else:  # networkx >= 3.3
    _louvain_communities.__name__ = "louvain_communities"
    louvain_communities = not_implemented_for("directed")(
        _louvain_decorator(_louvain_communities)
    )

    @louvain_communities._can_run
    def _(
        G,
        weight="weight",
        resolution=1,
        threshold=0.0000001,
        max_level=None,
        seed=None,
        *,
        dtype=None,
    ):
        # NetworkX allows both directed and undirected, but hipgraph only undirected.
        return not G.is_directed()
