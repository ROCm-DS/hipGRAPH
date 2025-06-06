# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import networkx as nx
import numpy as np
import pylibhipgraph as plc
from nx_hipgraph.convert import _to_graph
from nx_hipgraph.utils import (
    _dtype_param,
    _get_float_dtype,
    networkx_algorithm,
    not_implemented_for,
)

__all__ = ["katz_centrality"]


@not_implemented_for("multigraph")
@networkx_algorithm(
    extra_params=_dtype_param,
    is_incomplete=True,  # nstart and normalized=False not supported
    version_added="23.12",
    _plc="katz_centrality",
)
def katz_centrality(
    G,
    alpha=0.1,
    beta=1.0,
    max_iter=1000,
    tol=1.0e-6,
    nstart=None,
    normalized=True,
    weight=None,
    *,
    dtype=None,
):
    """`nstart` isn't used (but is checked), and `normalized=False` is not supported."""
    if not normalized:
        # Redundant with the `_can_run` check below when being dispatched by NetworkX,
        # but we raise here in case this funcion is called directly.
        raise NotImplementedError("normalized=False is not supported.")
    G = _to_graph(G, weight, 1, np.float32)
    if (N := len(G)) == 0:
        return {}
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    if nstart is not None:
        # Check if given nstart is valid even though we don't use it
        nstart = G._dict_to_nodearray(nstart, 0, dtype)
    b = bs = None
    try:
        b = float(beta)
    except (TypeError, ValueError) as exc:
        try:
            bs = G._dict_to_nodearray(beta, dtype=dtype)
            b = 1.0  # float value must be given to PLC (and will be ignored)
        except (KeyError, ValueError):
            raise nx.NetworkXError(
                "beta dictionary must have a value for every node"
            ) from exc
    try:
        node_ids, values = plc.katz_centrality(
            resource_handle=plc.ResourceHandle(),
            graph=G._get_plc_graph(weight, 1, dtype, store_transposed=True),
            betas=bs,
            alpha=alpha,
            beta=b,
            epsilon=N * tol,
            max_iterations=max_iter,
            do_expensive_check=False,
        )
    except RuntimeError as exc:
        # Errors from PLC are sometimes a little scary and not very helpful
        raise nx.PowerIterationFailedConvergence(max_iter) from exc
    return G._nodearrays_to_dict(node_ids, values)


@katz_centrality._can_run
def _(
    G,
    alpha=0.1,
    beta=1.0,
    max_iter=1000,
    tol=1.0e-6,
    nstart=None,
    normalized=True,
    weight=None,
    *,
    dtype=None,
):
    return normalized
