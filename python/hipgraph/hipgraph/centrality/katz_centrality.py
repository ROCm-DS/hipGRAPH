# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import warnings

import cudf
from hipgraph.utilities import df_score_to_dictionary, ensure_hipgraph_obj_for_nx
from pylibhipgraph import ResourceHandle
from pylibhipgraph import katz_centrality as pylibhipgraph_katz


def katz_centrality(
    G, alpha=None, beta=1.0, max_iter=100, tol=1.0e-6, nstart=None, normalized=True
):
    """
    Compute the Katz centrality for the nodes of the graph G. This
    implementation is based on a relaxed version of Katz defined by Foster
    with a reduced computational complexity of O(n+m)

    On a directed graph, hipGRAPH computes the out-edge Katz centrality score.
    This is opposite of NetworkX which compute the in-edge Katz centrality
    score by default.  You can flip the NetworkX edges, using G.reverse,
    so that the results match hipGRAPH.

    References
    ----------
    Foster, K.C., Muth, S.Q., Potterat, J.J. et al.
    Computational & Mathematical Organization Theory (2001) 7: 275.
    https://doi.org/10.1023/A:1013470632383

    Katz, L. (1953). A new status index derived from sociometric analysis.
    Psychometrika, 18(1), 39-43.

    Parameters
    ----------
    G : hipGRAPH.Graph or networkx.Graph
        hipGRAPH graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    alpha : float, optional (default=None)
        Attenuation factor defaulted to None. If alpha is not specified then
        it is internally calculated as 1/(degree_max) where degree_max is the
        maximum out degree.

        NOTE:
            The maximum acceptable value of alpha for convergence
            alpha_max = 1/(lambda_max) where lambda_max is the largest
            eigenvalue of the graph.
            Since lambda_max is always lesser than or equal to degree_max for a
            graph, alpha_max will always be greater than or equal to
            (1/degree_max). Therefore, setting alpha to (1/degree_max) will
            guarantee that it will never exceed alpha_max thus in turn
            fulfilling the requirement for convergence.

    beta : float, optional (default=None)
        Weight scalar added to each vertex's new Katz Centrality score in every
        iteration. If beta is not specified then it is set as 1.0.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.

    tol : float, optional (default=1.0e-6)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, hipGRAPH will use the default value which is 1.0e-6.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 1e-2 and 1e-6 are
        acceptable.

    nstart : cudf.Dataframe, optional (default=None)
        GPU Dataframe containing the initial guess for katz centrality.

        nstart['vertex'] : cudf.Series
            Contains the vertex identifiers
        nstart['values'] : cudf.Series
            Contains the katz centrality values of vertices

    normalized : not supported
        If True normalize the resulting katz centrality values

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding katz centrality values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['katz_centrality'] : cudf.Series
            Contains the katz centrality of vertices

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> kc = hipgraph.katz_centrality(G)

    """
    G, isNx = ensure_hipgraph_obj_for_nx(G, store_transposed=True)

    if G.store_transposed is False:
        warning_msg = (
            "Katz centrality expects the 'store_transposed' flag "
            "to be set to 'True' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    if alpha is None:
        degree_max = G.degree()["degree"].max()
        alpha = 1 / (degree_max)

    if (alpha is not None) and (alpha <= 0.0):
        raise ValueError(f"'alpha' must be a positive float or None, " f"got: {alpha}")

    elif (not isinstance(beta, float)) or (beta <= 0.0):
        raise ValueError(f"'beta' must be a positive float or None, " f"got: {beta}")
    if (not isinstance(max_iter, int)) or (max_iter <= 0):
        raise ValueError(f"'max_iter' must be a positive integer" f", got: {max_iter}")
    if (not isinstance(tol, float)) or (tol <= 0.0):
        raise ValueError(f"'tol' must be a positive float, got: {tol}")

    if nstart is not None:
        if G.renumbered is True:
            if len(G.renumber_map.implementation.col_names) > 1:
                cols = nstart.columns[:-1].to_list()
            else:
                cols = "vertex"
            nstart = G.add_internal_vertex_id(nstart, "vertex", cols)
            nstart = nstart[nstart.columns[0]]

    vertices, values = pylibhipgraph_katz(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        betas=nstart,
        alpha=alpha,
        beta=beta,
        epsilon=tol,
        max_iterations=max_iter,
        do_expensive_check=False,
    )

    vertices = cudf.Series(vertices)
    values = cudf.Series(values)

    df = cudf.DataFrame()
    df["vertex"] = vertices
    df["katz_centrality"] = values

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        dict = df_score_to_dictionary(df, "katz_centrality")
        return dict
    else:
        return df
