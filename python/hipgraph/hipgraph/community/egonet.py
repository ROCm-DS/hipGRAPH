# Copyright (c) 2021-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import warnings

import cudf
from hipgraph.utilities import ensure_hipgraph_obj, hipgraph_to_nx, is_nx_graph_type
from pylibhipgraph import ResourceHandle
from pylibhipgraph import ego_graph as pylibhipgraph_ego_graph


def _convert_graph_to_output_type(G, input_type):
    """
    Given a hipgraph.Graph, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    if is_nx_graph_type(input_type):
        return hipgraph_to_nx(G)

    else:
        return G


def _convert_df_series_to_output_type(df, offsets, input_type):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    if is_nx_graph_type(input_type):
        return df.to_pandas(), offsets.values_host.tolist()

    else:
        return df, offsets


# TODO: add support for a 'batch-mode' option.
def ego_graph(G, n, radius=1, center=True, undirected=None, distance=None):
    """
    Compute the induced subgraph of neighbors centered at node n,
    within a given radius.

    Parameters
    ----------
    G : hipgraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    n : integer or list, cudf.Series, cudf.DataFrame
        A single node as integer or a cudf.DataFrame if nodes are
        represented with multiple columns. If a cudf.DataFrame is provided,
        only the first row is taken as the node input.

    radius: integer, optional (default=1)
        Include all neighbors of distance<=radius from n.

    center: bool, optional
        Defaults to True. False is not supported

    undirected: bool, optional
        This parameter is here for NetworkX compatibility and is ignored

    distance: key, optional (default=None)
        This parameter is here for NetworkX compatibility and is ignored

    Returns
    -------
    G_ego : hipGRAPH.Graph or networkx.Graph
        A graph descriptor with a minimum spanning tree or forest.
        The networkx graph will not have all attributes copied over

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> ego_graph = hipgraph.ego_graph(G, 1, radius=2)

    """
    (G, input_type) = ensure_hipgraph_obj(G, nx_weight_attr="weight")

    result_graph = type(G)(directed=G.is_directed())

    if undirected is not None:
        warning_msg = (
            "The parameter 'undirected' is deprecated and "
            "will be removed in the next release"
        )
        warnings.warn(warning_msg, PendingDeprecationWarning)

    if isinstance(n, (int, list)):
        n = cudf.Series(n)
    if isinstance(n, cudf.Series):
        if G.renumbered is True:
            n = G.lookup_internal_vertex_id(n)
    elif isinstance(n, cudf.DataFrame):
        if G.renumbered is True:
            n = G.lookup_internal_vertex_id(n, n.columns)
    else:
        raise TypeError(
            f"'n' must be either an integer or a list or a cudf.Series"
            f" or a cudf.DataFrame, got: {type(n)}"
        )

    # Match the seed to the vertex dtype
    n_type = G.edgelist.edgelist_df["src"].dtype
    # FIXME: 'n' should represent a single vertex, but is not being verified
    n = n.astype(n_type)
    do_expensive_check = False

    source, destination, weight, _ = pylibhipgraph_ego_graph(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        source_vertices=n,
        radius=radius,
        do_expensive_check=do_expensive_check,
    )

    df = cudf.DataFrame()
    df["src"] = source
    df["dst"] = destination
    if weight is not None:
        df["weight"] = weight

    if G.renumbered:
        df, src_names = G.unrenumber(df, "src", get_column_names=True)
        df, dst_names = G.unrenumber(df, "dst", get_column_names=True)
    else:
        # FIXME: The original 'src' and 'dst' are not stored in 'simpleGraph'
        src_names = "src"
        dst_names = "dst"

    if G.edgelist.weights:
        result_graph.from_cudf_edgelist(
            df, source=src_names, destination=dst_names, edge_attr="weight"
        )
    else:
        result_graph.from_cudf_edgelist(df, source=src_names, destination=dst_names)
    return _convert_graph_to_output_type(result_graph, input_type)


def batched_ego_graphs(G, seeds, radius=1, center=True, undirected=None, distance=None):
    """
    This function is deprecated.

    Deprecated since 24.04. Batched support for multiple seeds will be added
    to `ego_graph`.

    Compute the induced subgraph of neighbors for each node in seeds
    within a given radius.

    Parameters
    ----------
    G : hipgraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    seeds : cudf.Series or list or cudf.DataFrame
        Specifies the seeds of the induced egonet subgraphs.

    radius: integer, optional (default=1)
        Include all neighbors of distance<=radius from n.

    center: bool, optional
        Defaults to True. False is not supported

    undirected: bool, optional
        Defaults to False. True is not supported

    distance: key, optional (default=None)
        Distances are counted in hops from n. Other cases are not supported.

    Returns
    -------
    ego_edge_lists : cudf.DataFrame or pandas.DataFrame
        GPU data frame containing all induced sources identifiers,
        destination identifiers, edge weights
    seeds_offsets: cudf.Series
        Series containing the starting offset in the returned edge list
        for each seed.

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> hipgraph.batched_ego_graphs(G, seeds=[1,5], radius=2)  # doctest: +SKIP
    """
    warning_msg = "This function is deprecated. Batched support for multiple vertices \
         will be added to `ego_graph`"
    warnings.warn(warning_msg, DeprecationWarning)

    (G, input_type) = ensure_hipgraph_obj(G, nx_weight_attr="weight")

    if seeds is not None:
        if isinstance(seeds, int):
            seeds = [seeds]
        if isinstance(seeds, list):
            seeds = cudf.Series(seeds)

        if G.renumbered is True:
            if isinstance(seeds, cudf.DataFrame):
                seeds = G.lookup_internal_vertex_id(seeds, seeds.columns)
            else:
                seeds = G.lookup_internal_vertex_id(seeds)

    # Match the seed to the vertex dtype
    seeds_type = G.edgelist.edgelist_df["src"].dtype
    seeds = seeds.astype(seeds_type)

    do_expensive_check = False
    source, destination, weight, offset = pylibhipgraph_ego_graph(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        source_vertices=seeds,
        radius=radius,
        do_expensive_check=do_expensive_check,
    )

    offsets = cudf.Series(offset)

    df = cudf.DataFrame()
    df["src"] = source
    df["dst"] = destination
    df["weight"] = weight

    if G.renumbered:
        df = G.unrenumber(df, "src", preserve_order=True)
        df = G.unrenumber(df, "dst", preserve_order=True)

    return _convert_df_series_to_output_type(df, offsets, input_type)
