# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
import numpy as np
from hipgraph.structure import Graph, MultiGraph
from hipgraph.utilities import cupy_package as cp
from hipgraph.utilities import (
    ensure_hipgraph_obj,
    is_cp_matrix_type,
    is_matrix_type,
    is_nx_graph_type,
)
from pylibhipgraph import ResourceHandle
from pylibhipgraph import sssp as pylibhipgraph_sssp


def _ensure_args(
    G, source, method, directed, return_predecessors, unweighted, overwrite, indices
):
    """
    Ensures the args passed in are usable for the API api_name and returns the
    args with proper defaults if not specified, or raises TypeError or
    ValueError if incorrectly specified.
    """
    # checks common to all input types
    if (method is not None) and (method != "auto"):
        raise ValueError("only 'auto' is currently accepted for method")
    if (indices is not None) and (type(indices) is list):
        raise ValueError("indices currently cannot be a list-like type")
    if (indices is not None) and (source is not None):
        raise TypeError("cannot specify both 'source' and 'indices'")
    if (indices is None) and (source is None):
        raise TypeError("must specify 'source' or 'indices', but not both")

    G_type = type(G)
    # Check for Graph-type inputs
    if G_type is Graph or is_nx_graph_type(G_type):
        # FIXME: Improve Graph-type checking
        exc_value = "'%s' cannot be specified for a Graph-type input"
        if directed is not None:
            raise TypeError(exc_value % "directed")
        if return_predecessors is not None:
            raise TypeError(exc_value % "return_predecessors")
        if unweighted is not None:
            raise TypeError(exc_value % "unweighted")
        if overwrite is not None:
            raise TypeError(exc_value % "overwrite")

        # Ensure source vertex is valid
        invalid_vertex_err = ValueError(
            f"Vertex {source} is not valid for the NetworkX Graph"
        )
        if is_nx_graph_type(G_type) and source not in G:
            raise invalid_vertex_err
        elif indices is None and not G.has_node(source):
            raise invalid_vertex_err

        directed = False

    # Check for non-Graph-type inputs
    else:
        if (directed is not None) and (type(directed) is not bool):
            raise ValueError("'directed' must be a bool")
        if (return_predecessors is not None) and (
            type(return_predecessors) is not bool
        ):
            raise ValueError("'return_predecessors' must be a bool")
        if (unweighted is not None) and (unweighted is not True):
            raise ValueError("'unweighted' currently must be True if " "specified")
        if (overwrite is not None) and (overwrite is not False):
            raise ValueError("'overwrite' currently must be False if " "specified")

    source = source if source is not None else indices
    if return_predecessors is None:
        return_predecessors = True

    return (source, directed, return_predecessors)


def _convert_df_to_output_type(df, input_type, return_predecessors):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.

    return_predecessors is only used for return values from cupy/scipy input
    types.
    """
    if input_type in [Graph, MultiGraph]:
        return df

    elif is_nx_graph_type(input_type):
        return df.to_pandas()

    elif is_matrix_type(input_type):
        # A CuPy/SciPy input means the return value will be a 2-tuple of:
        #   distance: cupy.ndarray
        #   predecessor: cupy.ndarray
        sorted_df = df.sort_values("vertex")
        if return_predecessors:
            if is_cp_matrix_type(input_type):
                return (
                    cp.from_dlpack(sorted_df["distance"].to_dlpack()),
                    cp.from_dlpack(sorted_df["predecessor"].to_dlpack()),
                )
            else:
                return (
                    sorted_df["distance"].to_numpy(),
                    sorted_df["predecessor"].to_numpy(),
                )
        else:
            if is_cp_matrix_type(input_type):
                return cp.from_dlpack(sorted_df["distance"].to_dlpack())
            else:
                return sorted_df["distance"].to_numpy()
    else:
        raise TypeError(f"input type {input_type} is not a supported type.")


def sssp(
    G,
    source=None,
    method=None,
    directed=None,
    return_predecessors=None,
    unweighted=None,
    overwrite=None,
    indices=None,
    cutoff=None,
    edge_attr="weight",
):
    """
    Compute the distance and predecessors for shortest paths from the specified
    source to all the vertices in the graph. The distances column will store
    the distance from the source to each vertex. The predecessors column will
    store each vertex's predecessor in the shortest path. Vertices that are
    unreachable will have a distance of infinity denoted by the maximum value
    of the data type and the predecessor set as -1. The source vertex's
    predecessor is also set to -1. Graphs with negative weight cycles are not
    supported.  Unweighted graphs are also unsupported.

    For finding shortest paths on an unweighted graph, use BFS instead.

    Parameters
    ----------
    graph : hipgraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix Graph or
        matrix object, which should contain the connectivity information. Edge
        weights, if present, should be single or double precision floating
        point values.
        The current implementation only supports weighted graphs.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    source : int
        Index of the source vertex.
    cutoff : double, optional (default=None)
        Maximum edge weight sum considered by the algorithm
    edge_attr : str, optional (default='weight')
        The name of the edge attribute that represents the weight of an edge.
        This currently applies only when G is a NetworkX Graph.
        Default value is 'weight', which follows NetworkX convention.

    Returns
    -------
    Return value type is based on the input type.  If G is a hipgraph.Graph,
    returns:

       cudf.DataFrame
          df['vertex']
              vertex id

          df['distance']
              gives the path distance from the starting vertex

          df['predecessor']
              the vertex it was reached from

    If G is a networkx.Graph, returns:

       pandas.DataFrame with contents equivalent to the cudf.DataFrame
       described above.

    If G is a CuPy or SciPy matrix, returns:
       a 2-tuple of CuPy ndarrays (if CuPy matrix input) or Numpy ndarrays (if
       SciPy matrix input) representing:

       distance: cupy or numpy ndarray
          ndarray of shortest distances between source and vertex.

       predecessor: cupy or numpy ndarray
          ndarray of predecessors of a vertex on the path from source, which
          can be used to reconstruct the shortest paths.

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> distances = hipgraph.sssp(G, 0)
    >>> distances
            distance  vertex  predecessor
    ...       ...     ...         ...
    ...       ...     ...         ...
    ...       ...     ...         ...

    """
    (source, directed, return_predecessors) = _ensure_args(
        G, source, method, directed, return_predecessors, unweighted, overwrite, indices
    )

    (G, input_type) = ensure_hipgraph_obj(
        G, nx_weight_attr=edge_attr, matrix_graph_type=Graph(directed=directed)
    )

    if not G.is_weighted():
        err_msg = (
            "'SSSP' requires the input graph to be weighted."
            "'BFS' should be used instead of 'SSSP' for unweighted graphs."
        )
        raise RuntimeError(err_msg)

    if not G.has_node(source):
        raise ValueError("Graph does not contain source vertex")

    if G.renumbered:
        if isinstance(source, cudf.DataFrame):
            source = G.lookup_internal_vertex_id(source, source.columns).iloc[0]
        else:
            source = G.lookup_internal_vertex_id(cudf.Series([source]))[0]

    if cutoff is None:
        cutoff = np.inf

    # compute_predecessors MUST be true in the current version of sssp
    vertices, distances, predecessors = pylibhipgraph_sssp(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        source=source,
        cutoff=cutoff,
        compute_predecessors=True,
        do_expensive_check=False,
    )

    df = cudf.DataFrame(
        {
            "distance": cudf.Series(distances),
            "vertex": cudf.Series(vertices),
            "predecessor": cudf.Series(predecessors),
        }
    )

    if G.renumbered:
        df = G.unrenumber(df, "vertex")
        df = G.unrenumber(df, "predecessor")
        df.fillna(-1, inplace=True)

    return _convert_df_to_output_type(df, input_type, return_predecessors)


def filter_unreachable(df):
    """
    Remove unreachable vertices from the result of SSSP or BFS

    Parameters
    ----------
    df : cudf.DataFrame
        cudf.DataFrame that is the output of SSSP or BFS

    Returns
    -------
    df : filtered cudf.DataFrame with only reachable vertices
        df['vertex'][i] gives the vertex id of the i'th vertex.
        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex.
        df['predecessor'][i] gives the vertex that was reached before the i'th
        vertex in the traversal.
    """
    if "distance" not in df:
        raise KeyError("No distance column found in input data frame")
    if np.issubdtype(df["distance"].dtype, np.integer):
        max_val = np.iinfo(df["distance"].dtype).max
        return df[df.distance != max_val]
    elif np.issubdtype(df["distance"].dtype, np.inexact):
        max_val = np.finfo(df["distance"].dtype).max
        return df[df.distance != max_val]
    else:
        raise TypeError("distance type unsupported")


def shortest_path(
    G,
    source=None,
    method=None,
    directed=None,
    return_predecessors=None,
    unweighted=None,
    overwrite=None,
    indices=None,
):
    """
    Alias for sssp(), provided for API compatibility with NetworkX. See sssp()
    for details.
    """
    return sssp(
        G, source, method, directed, return_predecessors, unweighted, overwrite, indices
    )


def shortest_path_length(G, source, target=None):
    """
    Compute the distance from a source vertex to one or all vertexes in graph.
    Uses Single Source Shortest Path (SSSP).

    Parameters
    ----------
    graph : hipGRAPH.Graph, NetworkX.Graph, or CuPy sparse COO matrix
        hipGRAPH graph descriptor with connectivity information. Edge weights,
        if present, should be single or double precision floating point values.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    source : Dependant on graph type. Index of the source vertex.

    If graph is an instance of hipGRAPH.Graph or CuPy sparse COO matrix:
        int

    If graph is an instance of a NetworkX.Graph:
        str

    target: Dependant on graph type. Vertex to find distance to.

    If graph is an instance of hipGRAPH.Graph or CuPy sparse COO matrix:
        int

    If graph is an instance of a NetworkX.Graph:
        str

    Returns
    -------
    Return value type is based on the input type.

    If target is None, returns:

        cudf.DataFrame
            df['vertex']
                vertex id

            df['distance']
                gives the path distance from the starting vertex

    If target is not None, returns:

        Distance from source to target vertex.
    """

    # verify target is in graph before traversing
    if target is not None:
        if not hasattr(G, "has_node"):
            # G is a cupy coo_matrix. Extract maximum possible vertex value
            as_matrix = G.toarray()
            if target < 0 or target >= max(as_matrix.shape[0], as_matrix.shape[1]):
                raise ValueError("Graph does not contain target vertex")
        elif not G.has_node(target):
            # G is an instance of hipgraph or networkx graph
            raise ValueError("Graph does not contain target vertex")

    df = sssp(G, source)

    if isinstance(df, tuple):
        # cupy path, df is tuple of (distance, predecessor)
        if target:
            return df[0][target - 1]
        results = cudf.DataFrame()
        results["vertex"] = range(df[0].shape[0])
        results["distance"] = df[0]
        return results

    else:
        # hipgraph and networkx path
        if target:
            target_distance = df.loc[df["vertex"] == target]
            return target_distance.iloc[0]["distance"]

        results = cudf.DataFrame()
        results["vertex"] = df["vertex"]
        results["distance"] = df["distance"]
        return results
