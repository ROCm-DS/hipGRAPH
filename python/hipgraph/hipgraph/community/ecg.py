# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import warnings

import cudf
from hipgraph.utilities import df_score_to_dictionary, ensure_hipgraph_obj_for_nx
from pylibhipgraph import ResourceHandle
from pylibhipgraph import ecg as pylibhipgraph_ecg


def ecg(
    input_graph,
    min_weight: float = 0.0001,
    ensemble_size: int = 100,
    max_level: int = 10,
    threshold: float = 1e-7,
    resolution: float = 1.0,
    random_state: int = None,
    weight=None,
):
    """
    Compute the Ensemble Clustering for Graphs (ECG) partition of the input
    graph. ECG runs truncated Louvain on an ensemble of permutations of the
    input graph, then uses the ensemble partitions to determine weights for
    the input graph. The final result is found by running full Louvain on
    the input graph using the determined weights.

    See https://arxiv.org/abs/1809.05578 for further information.

    Parameters
    ----------
    input_graph : hipgraph.Graph or NetworkX Graph
        The graph descriptor should contain the connectivity information
        and weights. The adjacency list will be computed if not already
        present.

    min_weight : float, optional (default=0.5)
        The minimum value to assign as an edgeweight in the ECG algorithm.
        It should be a value in the range [0,1] usually left as the default
        value of .05

    ensemble_size : integer, optional (default=16)
        The number of graph permutations to use for the ensemble.
        The default value is 16, larger values may produce higher quality
        partitions for some graphs.

    max_level : integer, optional (default=100)
        This controls the maximum number of levels/iterations of the ECG
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    threshold: float
        Modularity gain threshold for each level. If the gain of
        modularity between 2 levels of the algorithm is less than the
        given threshold then the algorithm stops and returns the
        resulting communities.
        Defaults to 1e-7.

    resolution: float, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

    random_state: int, optional(default=None)
        Random state to use when generating samples.  Optional argument,
        defaults to a hash of process id, time, and hostname.

    weight : str, optional (default=None)
        Deprecated.
        This parameter is here for NetworkX compatibility and
        represents which NetworkX data column represents Edge weights.

    Returns
    -------
    parts : cudf.DataFrame or python dictionary
        GPU data frame of size V containing two columns, the vertex id and
        the partition id it is assigned to.

        parts[vertex] : cudf.Series
            Contains the vertex identifiers
        parts[partition] : cudf.Series
            Contains the partition assigned to the vertices

    modularity_score : float
        A floating point number containing the global modularity score
        of the partitioning.

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> parts, mod = hipgraph.ecg(G)

    """

    input_graph, isNx = ensure_hipgraph_obj_for_nx(input_graph)

    if isNx:
        warning_msg = (
            " We are deprecating support for handling "
            "NetworkX types in the next release."
        )
        warnings.warn(warning_msg, UserWarning)

    if weight is not None:
        warning_msg = (
            "This parameter is deprecated and will be removed in the next release."
        )
        warnings.warn(warning_msg, UserWarning)

    vertex, partition, modularity_score = pylibhipgraph_ecg(
        resource_handle=ResourceHandle(),
        random_state=random_state,
        graph=input_graph._plc_graph,
        min_weight=min_weight,
        ensemble_size=ensemble_size,
        max_level=max_level,
        threshold=threshold,
        resolution=resolution,
        do_expensive_check=False,
    )

    parts = cudf.DataFrame()
    parts["vertex"] = vertex
    parts["partition"] = partition

    if input_graph.renumbered:
        parts = input_graph.unrenumber(parts, "vertex")

    if isNx is True:
        parts = df_score_to_dictionary(parts, "partition")

    return parts, modularity_score
