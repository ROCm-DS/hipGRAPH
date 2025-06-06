# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
import numpy as np
from hipgraph.utilities import df_score_to_dictionary, ensure_hipgraph_obj_for_nx
from pylibhipgraph import ResourceHandle
from pylibhipgraph import (
    analyze_clustering_edge_cut as pylibhipgraph_analyze_clustering_edge_cut,
)
from pylibhipgraph import (
    analyze_clustering_modularity as pylibhipgraph_analyze_clustering_modularity,
)
from pylibhipgraph import (
    analyze_clustering_ratio_cut as pylibhipgraph_analyze_clustering_ratio_cut,
)
from pylibhipgraph import (
    balanced_cut_clustering as pylibhipgraph_balanced_cut_clustering,
)
from pylibhipgraph import (
    spectral_modularity_maximization as pylibhipgraph_spectral_modularity_maximization,
)


def spectralBalancedCutClustering(
    G,
    num_clusters,
    num_eigen_vects=2,
    evs_tolerance=0.00001,
    evs_max_iter=100,
    kmean_tolerance=0.00001,
    kmean_max_iter=100,
):
    """
    Compute a clustering/partitioning of the given graph using the spectral
    balanced cut method.

    Parameters
    ----------
    G : hipgraph.Graph or networkx.Graph
        Graph descriptor

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    num_clusters : integer
        Specifies the number of clusters to find, must be greater than 1

    num_eigen_vects : integer, optional
        Specifies the number of eigenvectors to use. Must be lower or equal to
        num_clusters. Default is 2

    evs_tolerance: float, optional
        Specifies the tolerance to use in the eigensolver.
        Default is 0.00001

    evs_max_iter: integer, optional
        Specifies the maximum number of iterations for the eigensolver.
        Default is 100

    kmean_tolerance: float, optional
        Specifies the tolerance to use in the k-means solver.
        Default is 0.00001

    kmean_max_iter: integer, optional
        Specifies the maximum number of iterations for the k-means solver.
        Default is 100

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding cluster assignments.

        df['vertex'] : cudf.Series
            contains the vertex identifiers
        df['cluster'] : cudf.Series
            contains the cluster assignments

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> df = hipgraph.spectralBalancedCutClustering(G, 5)

    """

    # Error checking in C++ code

    G, isNx = ensure_hipgraph_obj_for_nx(G)
    # Check if vertex type is "int32"
    if (
        G.edgelist.edgelist_df.dtypes.iloc[0] != np.int32
        or G.edgelist.edgelist_df.dtypes.iloc[1] != np.int32
    ):
        raise ValueError(
            "'spectralBalancedCutClustering' requires the input graph's vertex to be "
            "of type 'int32'"
        )
    vertex, partition = pylibhipgraph_balanced_cut_clustering(
        ResourceHandle(),
        G._plc_graph,
        num_clusters,
        num_eigen_vects,
        evs_tolerance,
        evs_max_iter,
        kmean_tolerance,
        kmean_max_iter,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["cluster"] = partition

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        df = df_score_to_dictionary(df, "cluster")

    return df


def spectralModularityMaximizationClustering(
    G,
    num_clusters,
    num_eigen_vects=2,
    evs_tolerance=0.00001,
    evs_max_iter=100,
    kmean_tolerance=0.00001,
    kmean_max_iter=100,
):
    """
    Compute a clustering/partitioning of the given graph using the spectral
    modularity maximization method.

    Parameters
    ----------
    G : hipgraph.Graph or networkx.Graph
        hipGRAPH graph descriptor. This graph should have edge weights.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    num_clusters : integer
        Specifies the number of clusters to find

    num_eigen_vects : integer, optional
        Specifies the number of eigenvectors to use. Must be lower or equal to
        num_clusters.  Default is 2

    evs_tolerance: float, optional
        Specifies the tolerance to use in the eigensolver.
        Default is 0.00001

    evs_max_iter: integer, optional
        Specifies the maximum number of iterations for the eigensolver.
        Default is 100

    kmean_tolerance: float, optional
        Specifies the tolerance to use in the k-means solver.
        Default is 0.00001

    kmean_max_iter: integer, optional
        Specifies the maximum number of iterations for the k-means solver.
        Default is 100

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding cluster assignments.

        df['vertex'] : cudf.Series
            contains the vertex identifiers
        df['cluster'] : cudf.Series
            contains the cluster assignments

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> df = hipgraph.spectralModularityMaximizationClustering(G, 5)

    """

    G, isNx = ensure_hipgraph_obj_for_nx(G)
    if (
        G.edgelist.edgelist_df.dtypes.iloc[0] != np.int32
        or G.edgelist.edgelist_df.dtypes.iloc[1] != np.int32
    ):
        raise ValueError(
            "'spectralModularityMaximizationClustering' requires the input graph's "
            "vertex to be of type 'int32'"
        )

    vertex, partition = pylibhipgraph_spectral_modularity_maximization(
        ResourceHandle(),
        G._plc_graph,
        num_clusters,
        num_eigen_vects,
        evs_tolerance,
        evs_max_iter,
        kmean_tolerance,
        kmean_max_iter,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["cluster"] = partition

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        df = df_score_to_dictionary(df, "cluster")

    return df


def analyzeClustering_modularity(
    G, n_clusters, clustering, vertex_col_name="vertex", cluster_col_name="cluster"
):
    """
    Compute the modularity score for a given partitioning/clustering.
    The assumption is that “clustering” is the results from a call
    from a special clustering algorithm and contains columns named
    “vertex” and “cluster”.

    Parameters
    ----------
    G : hipgraph.Graph or networkx.Graph
        graph descriptor. This graph should have edge weights.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    n_clusters : integer
        Specifies the number of clusters in the given clustering

    clustering : cudf.DataFrame
        The cluster assignment to analyze.

    vertex_col_name : str or list of str, optional (default='vertex')
        The names of the column in the clustering dataframe identifying
        the external vertex id

    cluster_col_name : str, optional (default='cluster')
        The name of the column in the clustering dataframe identifying
        the cluster id

    Returns
    -------
    score : float
        The computed modularity score

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> df = hipgraph.spectralBalancedCutClustering(G, 5)
    >>> score = hipgraph.analyzeClustering_modularity(G, 5, df)

    """
    if type(vertex_col_name) is list:
        if not all(isinstance(name, str) for name in vertex_col_name):
            raise Exception("vertex_col_name must be list of string")
    elif type(vertex_col_name) is not str:
        raise Exception("vertex_col_name must be a string")

    if type(cluster_col_name) is not str:
        raise Exception("cluster_col_name must be a string")

    G, isNx = ensure_hipgraph_obj_for_nx(G)
    if (
        G.edgelist.edgelist_df.dtypes.iloc[0] != np.int32
        or G.edgelist.edgelist_df.dtypes.iloc[1] != np.int32
    ):
        raise ValueError(
            "'analyzeClustering_modularity' requires the input graph's "
            "vertex to be of type 'int32'"
        )

    if G.renumbered:
        clustering = G.add_internal_vertex_id(
            clustering, "vertex", vertex_col_name, drop=True
        )

    if clustering.dtypes.iloc[0] != np.int32 or clustering.dtypes.iloc[1] != np.int32:
        raise ValueError(
            "'analyzeClustering_modularity' requires both the clustering 'vertex' "
            "and 'cluster' to be of type 'int32'"
        )

    score = pylibhipgraph_analyze_clustering_modularity(
        ResourceHandle(),
        G._plc_graph,
        n_clusters,
        clustering["vertex"],
        clustering[cluster_col_name],
    )

    return score


def analyzeClustering_edge_cut(
    G, n_clusters, clustering, vertex_col_name="vertex", cluster_col_name="cluster"
):
    """
    Compute the edge cut score for a partitioning/clustering
    The assumption is that “clustering” is the results from a call
    from a special clustering algorithm and contains columns named
    “vertex” and “cluster”.

    Parameters
    ----------
    G : hipgraph.Graph
        hipGRAPH graph descriptor

    n_clusters : integer
        Specifies the number of clusters in the given clustering

    clustering : cudf.DataFrame
        The cluster assignment to analyze.

    vertex_col_name : str, optional (default='vertex')
        The name of the column in the clustering dataframe identifying
        the external vertex id

    cluster_col_name : str, optional (default='cluster')
        The name of the column in the clustering dataframe identifying
        the cluster id

    Returns
    -------
    score : float
        The computed edge cut score

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> df = hipgraph.spectralBalancedCutClustering(G, 5)
    >>> score = hipgraph.analyzeClustering_edge_cut(G, 5, df)

    """
    if type(vertex_col_name) is list:
        if not all(isinstance(name, str) for name in vertex_col_name):
            raise Exception("vertex_col_name must be list of string")
    elif type(vertex_col_name) is not str:
        raise Exception("vertex_col_name must be a string")

    if type(cluster_col_name) is not str:
        raise Exception("cluster_col_name must be a string")

    G, isNx = ensure_hipgraph_obj_for_nx(G)

    if (
        G.edgelist.edgelist_df.dtypes.iloc[0] != np.int32
        or G.edgelist.edgelist_df.dtypes.iloc[1] != np.int32
    ):
        raise ValueError(
            "'analyzeClustering_edge_cut' requires the input graph's vertex to be "
            "of type 'int32'"
        )

    if G.renumbered:
        clustering = G.add_internal_vertex_id(
            clustering, "vertex", vertex_col_name, drop=True
        )

    if clustering.dtypes.iloc[0] != np.int32 or clustering.dtypes.iloc[1] != np.int32:
        raise ValueError(
            "'analyzeClustering_edge_cut' requires both the clustering 'vertex' "
            "and 'cluster' to be of type 'int32'"
        )

    score = pylibhipgraph_analyze_clustering_edge_cut(
        ResourceHandle(),
        G._plc_graph,
        n_clusters,
        clustering["vertex"],
        clustering[cluster_col_name],
    )

    return score


def analyzeClustering_ratio_cut(
    G, n_clusters, clustering, vertex_col_name="vertex", cluster_col_name="cluster"
):
    """
    Compute the ratio cut score for a partitioning/clustering

    Parameters
    ----------
    G : hipgraph.Graph
        hipGRAPH graph descriptor. This graph should have edge weights.

    n_clusters : integer
        Specifies the number of clusters in the given clustering

    clustering : cudf.DataFrame
        The cluster assignment to analyze.

    vertex_col_name : str, optional (default='vertex')
        The name of the column in the clustering dataframe identifying
        the external vertex id

    cluster_col_name : str, optional (default='cluster')
        The name of the column in the clustering dataframe identifying
        the cluster id

    Returns
    -------
    score : float
        The computed ratio cut score

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> df = hipgraph.spectralBalancedCutClustering(G, 5)
    >>> score = hipgraph.analyzeClustering_ratio_cut(G, 5, df, 'vertex',
    ...                                             'cluster')

    """
    if type(vertex_col_name) is list:
        if not all(isinstance(name, str) for name in vertex_col_name):
            raise Exception("vertex_col_name must be list of string")
    elif type(vertex_col_name) is not str:
        raise Exception("vertex_col_name must be a string")

    if type(cluster_col_name) is not str:
        raise Exception("cluster_col_name must be a string")

    if G.renumbered:
        clustering = G.add_internal_vertex_id(
            clustering, "vertex", vertex_col_name, drop=True
        )

    if clustering.dtypes.iloc[0] != np.int32 or clustering.dtypes.iloc[1] != np.int32:
        raise ValueError(
            "'analyzeClustering_ratio_cut' requires both the clustering 'vertex' "
            "and 'cluster' to be of type 'int32'"
        )

    score = pylibhipgraph_analyze_clustering_ratio_cut(
        ResourceHandle(),
        G._plc_graph,
        n_clusters,
        clustering["vertex"],
        clustering[cluster_col_name],
    )

    return score
