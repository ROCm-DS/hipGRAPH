# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import warnings

import cudf
from hipgraph.utilities import df_score_to_dictionary, ensure_hipgraph_obj_for_nx
from pylibhipgraph import ResourceHandle
from pylibhipgraph import eigenvector_centrality as pylib_eigen


def eigenvector_centrality(G, max_iter=100, tol=1.0e-6):
    """
    Compute the eigenvector centrality for a graph G.

    Eigenvector centrality computes the centrality for a node based on the
    centrality of its neighbors. The eigenvector centrality for node i is the
    i-th element of the vector x defined by the eigenvector equation.

    Parameters
    ----------
    G : hipGRAPH.Graph or networkx.Graph
        hipGRAPH graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.

    tol : float, optional (default=1e-6)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, hipGRAPH will use the default value which is 1.0e-6.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 1e-2 and 1e-6 are
        acceptable.

    normalized : not supported
        If True normalize the resulting eigenvector centrality values

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding eigenvector centrality values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers

        df['eigenvector_centrality'] : cudf.Series
            Contains the eigenvector centrality of vertices

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> ec = hipgraph.eigenvector_centrality(G)

    """
    if (not isinstance(max_iter, int)) or max_iter <= 0:
        raise ValueError(f"'max_iter' must be a positive integer" f", got: {max_iter}")
    if (not isinstance(tol, float)) or (tol <= 0.0):
        raise ValueError(f"'tol' must be a positive float, got: {tol}")

    G, isNx = ensure_hipgraph_obj_for_nx(G, store_transposed=True)
    if G.store_transposed is False:
        warning_msg = (
            "Eigenvector centrality expects the 'store_transposed' "
            "flag to be set to 'True' for optimal performance "
            "during the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    vertices, values = pylib_eigen(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        epsilon=tol,
        max_iterations=max_iter,
        do_expensive_check=False,
    )

    vertices = cudf.Series(vertices)
    values = cudf.Series(values)

    df = cudf.DataFrame()
    df["vertex"] = vertices
    df["eigenvector_centrality"] = values

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        dict = df_score_to_dictionary(df, "eigenvector_centrality")
        return dict
    else:
        return df
