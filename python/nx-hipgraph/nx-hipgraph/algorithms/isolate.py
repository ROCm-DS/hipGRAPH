# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from nx_hipgraph.convert import _to_graph
from nx_hipgraph.utils import index_dtype, networkx_algorithm

if TYPE_CHECKING:  # pragma: no cover
    from nx_hipgraph.typing import IndexValue

__all__ = ["is_isolate", "isolates", "number_of_isolates"]


@networkx_algorithm(version_added="23.10")
def is_isolate(G, n):
    G = _to_graph(G)
    index = n if G.key_to_id is None else G.key_to_id[n]
    return not (
        (G.src_indices == index).any().tolist()
        or G.is_directed()
        and (G.dst_indices == index).any().tolist()
    )


@is_isolate._should_run
def _(G, n):
    return "Fast algorithm; not worth converting."


def _mark_isolates(G, symmetrize=None) -> cp.ndarray[bool]:
    """Return a boolean mask array indicating indices of isolated nodes."""
    mark_isolates = cp.ones(len(G), bool)
    if G.is_directed() and symmetrize == "intersection":
        N = G._N
        # Upcast to int64 so indices don't overflow
        src_dst = N * G.src_indices.astype(np.int64) + G.dst_indices
        src_dst_T = G.src_indices + N * G.dst_indices.astype(np.int64)
        src_dst_new = cp.intersect1d(src_dst, src_dst_T)
        new_indices = cp.floor_divide(src_dst_new, N, dtype=index_dtype)
        mark_isolates[new_indices] = False
    else:
        mark_isolates[G.src_indices] = False
        if G.is_directed():
            mark_isolates[G.dst_indices] = False
    return mark_isolates


def _isolates(G, symmetrize=None) -> cp.ndarray[IndexValue]:
    """Like isolates, but return an array of indices instead of an iterator of nodes."""
    G = _to_graph(G)
    return cp.nonzero(_mark_isolates(G, symmetrize=symmetrize))[0]


@networkx_algorithm(version_added="23.10")
def isolates(G):
    G = _to_graph(G)
    return G._nodeiter_to_iter(iter(_isolates(G).tolist()))


@networkx_algorithm(version_added="23.10")
def number_of_isolates(G):
    G = _to_graph(G)
    return int(cp.count_nonzero(_mark_isolates(G)))
