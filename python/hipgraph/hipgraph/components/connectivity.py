# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
from hipgraph.components import connectivity_wrapper
from hipgraph.structure import Graph
from hipgraph.utilities import cupy_package as cp
from hipgraph.utilities import (
    df_score_to_dictionary,
    ensure_hipgraph_obj,
    is_cp_matrix_type,
    is_matrix_type,
    is_nx_graph_type,
)
from pylibhipgraph import ResourceHandle
from pylibhipgraph import weakly_connected_components as pylibhipgraph_wcc


def _ensure_args(api_name, G, directed, connection, return_labels):
    """
    Ensures the args passed in are usable for the API api_name and returns the
    args with proper defaults if not specified, or raises TypeError or
    ValueError if incorrectly specified.
    """
    G_type = type(G)
    # Check for Graph-type inputs and set defaults if unset
    if (G_type in [Graph]) or is_nx_graph_type(G_type):
        exc_value = "'%s' cannot be specified for a Graph-type input"
        if directed is not None:
            raise TypeError(exc_value % "directed")
        if return_labels is not None:
            raise TypeError(exc_value % "return_labels")

        directed = True
        return_labels = True

    # Check for non-Graph-type inputs and set defaults if unset
    else:
        directed = True if (directed is None) else directed
        return_labels = True if (return_labels is None) else return_labels

    # Handle connection type, based on API being called
    if api_name == "strongly_connected_components":
        if (connection is not None) and (connection != "strong"):
            raise TypeError("'connection' must be 'strong' for " f"{api_name}()")
        connection = "strong"
    elif api_name == "weakly_connected_components":
        if (connection is not None) and (connection != "weak"):
            raise TypeError("'connection' must be 'weak' for " f"{api_name}()")
        connection = "weak"
    else:
        raise RuntimeError("invalid API name specified (internal): " f"{api_name}")

    return (directed, connection, return_labels)


def _convert_df_to_output_type(df, input_type, return_labels):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    return_labels is only used for return values from cupy/scipy input types.
    """
    if input_type in [Graph]:
        return df

    elif is_nx_graph_type(input_type):
        return df_score_to_dictionary(df, "labels", "vertex")

    elif is_matrix_type(input_type):
        # Convert DF of 2 columns (labels, vertices) to the SciPy-style return
        # value:
        #   n_components: int
        #       The number of connected components (number of unique labels).
        #   labels: ndarray
        #       The length-N array of labels of the connected components.
        n_components = df["labels"].nunique()
        sorted_df = df.sort_values("vertex")
        if return_labels:
            if is_cp_matrix_type(input_type):
                labels = cp.from_dlpack(sorted_df["labels"].to_dlpack())
            else:
                labels = sorted_df["labels"].to_numpy()
            return (n_components, labels)
        else:
            return n_components

    else:
        raise TypeError(f"input type {input_type} is not a supported type.")


def weakly_connected_components(G, directed=None, connection=None, return_labels=None):
    """
    Generate the Weakly Connected Components and attach a component label to
    each vertex.

    Parameters
    ----------
    G : hipgraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix

        Graph or matrix object, which should contain the connectivity
        information (edge weights are not used for this algorithm). If using a
        graph object, the graph must be undirected where an
        undirected edge is represented by a directed edge in both directions.
        The adjacency list will be computed if not already present. The number
        of vertices should fit into a 32b int.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    directed : bool, optional (default=None)

        NOTE
            For non-Graph-type (eg. sparse matrix) values of G only.
            Raises TypeError if used with a Graph object.

        If True, then convert the input matrix to a Graph(directed=True)
        and only move from point i to point j along paths csgraph[i, j]. If
        False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i].

    connection : str, optional (default=None)

        Added for SciPy compatibility, can only be specified for non-Graph-type
        (eg. sparse matrix) values of G only (raises TypeError if used with a
        Graph object), and can only be set to "weak" for this API.

    return_labels : bool, optional (default=True)

        NOTE
            For non-Graph-type (eg. sparse matrix) values of G only. Raises
            TypeError if used with a Graph object.

        If True, then return the labels for each of the connected
        components.

    Returns
    -------
    Return value type is based on the input type.  If G is a hipgraph.Graph,
    returns:

       cudf.DataFrame
           GPU data frame containing two cudf.Series of size V: the vertex
           identifiers and the corresponding component identifier.

           df['vertex']
               Contains the vertex identifier
           df['labels']
               The component identifier

    If G is a networkx.Graph, returns:

       python dictionary, where keys are vertices and values are the component
       identifiers.

    If G is a CuPy or SciPy matrix, returns:

       CuPy ndarray (if CuPy matrix input) or Numpy ndarray (if SciPy matrix
       input) of shape (<num vertices>, 2), where column 0 contains component
       identifiers and column 1 contains vertices.

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> df = hipgraph.weakly_connected_components(G)

    """
    (directed, connection, return_labels) = _ensure_args(
        "weakly_connected_components", G, directed, connection, return_labels
    )

    # FIXME: allow nx_weight_attr to be specified
    (G, input_type) = ensure_hipgraph_obj(
        G, nx_weight_attr="weight", matrix_graph_type=Graph(directed=directed)
    )

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    vertex, labels = pylibhipgraph_wcc(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        offsets=None,
        indices=None,
        weights=None,
        labels=None,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["labels"] = labels

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    return _convert_df_to_output_type(df, input_type, return_labels)


def strongly_connected_components(
    G, directed=None, connection=None, return_labels=None
):
    """
    Generate the Strongly Connected Components and attach a component label to
    each vertex.

    Parameters
    ----------
    G : hipgraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix

        Graph or matrix object, which should contain the connectivity
        information (edge weights are not used for this algorithm). If using a
        graph object, the graph can be either directed or undirected where an
        undirected edge is represented by a directed edge in both directions.
        The adjacency list will be computed if not already present.  The number
        of vertices should fit into a 32b int.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    directed : bool, optional (default=True)

        NOTE
            For non-Graph-type (eg. sparse matrix) values of G only.
            Raises TypeError if used with a Graph object.

        If True, then convert the input matrix to a Graph(directed=True)
        and only move from point i to point j along paths csgraph[i, j]. If
        False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i].

    connection : str, optional (default=None)

        Added for SciPy compatibility, can only be specified for non-Graph-type
        (eg. sparse matrix) values of G only (raises TypeError if used with a
        Graph object), and can only be set to "strong" for this API.

    return_labels : bool, optional (default=True)

        NOTE
            For non-Graph-type (eg. sparse matrix) values of G only. Raises
            TypeError if used with a Graph object.

        If True, then return the labels for each of the connected
        components.

    Returns
    -------
    Return value type is based on the input type.  If G is a hipgraph.Graph,
    returns:

       cudf.DataFrame
           GPU data frame containing two cudf.Series of size V: the vertex
           identifiers and the corresponding component identifier.

           df['vertex']
               Contains the vertex identifier
           df['labels']
               The component identifier

    If G is a networkx.Graph, returns:

       python dictionary, where keys are vertices and values are the component
       identifiers.

    If G is a CuPy or SciPy matrix, returns:

       CuPy ndarray (if CuPy matrix input) or Numpy ndarray (if SciPy matrix
       input) of shape (<num vertices>, 2), where column 0 contains component
       identifiers and column 1 contains vertices.

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> df = hipgraph.strongly_connected_components(G)

    """
    (directed, connection, return_labels) = _ensure_args(
        "strongly_connected_components", G, directed, connection, return_labels
    )

    # FIXME: allow nx_weight_attr to be specified
    (G, input_type) = ensure_hipgraph_obj(
        G, nx_weight_attr="weight", matrix_graph_type=Graph(directed=directed)
    )
    # Renumber the vertices so that they are contiguous (required)
    # FIXME: Remove 'renumbering' once the algo leverage the CAPI graph
    if not G.renumbered:
        edgelist = G.edgelist.edgelist_df
        renumbered_edgelist_df, renumber_map = G.renumber_map.renumber(
            edgelist, ["src"], ["dst"]
        )
        renumbered_src_col_name = renumber_map.renumbered_src_col_name
        renumbered_dst_col_name = renumber_map.renumbered_dst_col_name
        G.edgelist.edgelist_df = renumbered_edgelist_df.rename(
            columns={renumbered_src_col_name: "src", renumbered_dst_col_name: "dst"}
        )
        G.properties.renumbered = True
        G.renumber_map = renumber_map

    df = connectivity_wrapper.strongly_connected_components(G)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    return _convert_df_to_output_type(df, input_type, return_labels)


def connected_components(G, directed=None, connection="weak", return_labels=None):
    """
    Generate either the strongly or weakly connected components and attach a
    component label to each vertex.

    Parameters
    ----------
    G : hipgraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix

        Graph or matrix object, which should contain the connectivity
        information (edge weights are not used for this algorithm). If using a
        graph object, the graph can be either directed or undirected where an
        undirected edge is represented by a directed edge in both directions.
        The adjacency list will be computed if not already present.  The number
        of vertices should fit into a 32b int.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    directed : bool, optional (default=True)

        NOTE
            For non-Graph-type (eg. sparse matrix) values of G only. Raises
            TypeError if used with a Graph object.

        If True, then convert the input matrix to a Graph(directed=True)
        and only move from point i to point j along paths csgraph[i, j]. If
        False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i].

    connection : str, optional (default='weak')

        NOTE
            For Graph-type values of G, weak components are only
            supported for undirected graphs.

        [‘weak’|’strong’]. Return either weakly or strongly connected
        components.

    return_labels : bool, optional (default=True)

        NOTE
            For non-Graph-type (eg. sparse matrix) values of G only. Raises
            TypeError if used with a Graph object.

        If True, then return the labels for each of the connected
        components.

    Returns
    -------
    Return value type is based on the input type.  If G is a hipgraph.Graph,
    returns:

       cudf.DataFrame
           GPU data frame containing two cudf.Series of size V: the vertex
           identifiers and the corresponding component identifier.

           df['vertex']
               Contains the vertex identifier
           df['labels']
               The component identifier

    If G is a networkx.Graph, returns:

       python dictionary, where keys are vertices and values are the component
       identifiers.

    If G is a CuPy or SciPy matrix, returns:

       CuPy ndarray (if CuPy matrix input) or Numpy ndarray (if SciPy matrix
       input) of shape (<num vertices>, 2), where column 0 contains component
       identifiers and column 1 contains vertices.

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> df = hipgraph.connected_components(G, connection="weak")

    """
    if connection == "weak":
        return weakly_connected_components(G, directed, connection, return_labels)
    elif connection == "strong":
        return strongly_connected_components(G, directed, connection, return_labels)
    else:
        raise ValueError(
            f"invalid connection type: {connection}, "
            "must be either 'strong' or 'weak'"
        )
