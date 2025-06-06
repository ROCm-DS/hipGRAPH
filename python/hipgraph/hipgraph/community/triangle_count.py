# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import warnings

import cudf
from hipgraph.utilities import ensure_hipgraph_obj_for_nx
from pylibhipgraph import ResourceHandle
from pylibhipgraph import triangle_count as pylibhipgraph_triangle_count


# FIXME: Move this function to the utility module so that it can be
# shared by other algos
def ensure_valid_dtype(input_graph, start_list):
    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    if isinstance(start_list, cudf.Series):
        start_list_dtypes = start_list.dtype
    else:
        start_list_dtypes = start_list.dtypes[0]

    if start_list_dtypes != vertex_dtype:
        warning_msg = (
            "Triangle_count requires 'start_list' to match the graph's 'vertex' type. "
            f"input graph's vertex type is: {vertex_dtype} and got "
            f"'start_list' of type: {start_list_dtypes}."
        )
        warnings.warn(warning_msg, UserWarning)
        start_list = start_list.astype(vertex_dtype)

    return start_list


def triangle_count(G, start_list=None):
    """
    Compute the number of triangles (cycles of length three) in the
    input graph.

    Parameters
    ----------
    G : hipgraph.graph or networkx.Graph
        hipGRAPH graph descriptor, should contain the connectivity information,
        (edge weights are not used in this algorithm).
        The current implementation only supports undirected graphs.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    start_list : list or cudf.Series
        list of vertices for triangle count. if None the entire set of vertices
        in the graph is processed

    Returns
    -------
    result : cudf.DataFrame
        GPU data frame containing 2 cudf.Series

    ddf['vertex']: cudf.Series
            Contains the triangle counting vertices
    ddf['counts']: cudf.Series
        Contains the triangle counting counts

    Examples
    --------
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv',
    ...                     delimiter = ' ',
    ...                     dtype=['int32', 'int32', 'float32'],
    ...                     header=None)
    >>> G = hipgraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1', edge_attr='2')
    >>> count = hipgraph.triangle_count(G)

    """
    G, _ = ensure_hipgraph_obj_for_nx(G)

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    if start_list is not None:
        if isinstance(start_list, int):
            start_list = [start_list]
        if isinstance(start_list, list):
            start_list = cudf.Series(start_list)

        if not isinstance(start_list, cudf.Series):
            raise TypeError(
                f"'start_list' must be either a list or a cudf.Series,"
                f"got: {start_list.dtype}"
            )
        start_list = ensure_valid_dtype(G, start_list)

        if G.renumbered is True:
            if isinstance(start_list, cudf.DataFrame):
                start_list = G.lookup_internal_vertex_id(start_list, start_list.columns)
            else:
                start_list = G.lookup_internal_vertex_id(start_list)

    do_expensive_check = False

    vertex, counts = pylibhipgraph_triangle_count(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        start_list=start_list,
        do_expensive_check=do_expensive_check,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["counts"] = counts

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    return df
