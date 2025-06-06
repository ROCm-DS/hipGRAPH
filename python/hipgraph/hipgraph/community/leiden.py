# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from typing import Tuple, Union

import cudf
from hipgraph.structure import Graph
from hipgraph.utilities import df_score_to_dictionary, ensure_hipgraph_obj_for_nx
from hipgraph.utilities.utils import import_optional
from pylibhipgraph import ResourceHandle
from pylibhipgraph import leiden as pylibhipgraph_leiden

# FIXME: the networkx.Graph type used in the type annotation for
# leiden() is specified using a string literal to avoid depending on
# and importing networkx. Instead, networkx is imported optionally, which may
# cause a problem for a type checker if run in an environment where networkx is
# not installed.
networkx = import_optional("networkx")


def leiden(
    G: Union[Graph, "networkx.Graph"],
    max_iter: int = 100,
    resolution: float = 1.0,
    random_state: int = None,
    theta: int = 1.0,
) -> Tuple[cudf.DataFrame, float]:
    """
    Compute the modularity optimizing partition of the input graph using the
    Leiden algorithm

    It uses the Leiden method described in:

    Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden:
    guaranteeing well-connected communities. Scientific reports, 9(1), 5233.
    doi: 10.1038/s41598-019-41695-z

    Parameters
    ----------
    G : hipgraph.Graph
        hipGRAPH graph descriptor of type Graph

        The current implementation only supports undirected weighted graphs.

        The adjacency list will be computed if not already present.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    max_iter : integer, optional (default=100)
        This controls the maximum number of levels/iterations of the Leiden
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    resolution: float, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

    random_state: int, optional(default=None)
        Random state to use when generating samples.  Optional argument,
        defaults to a hash of process id, time, and hostname.

    theta: float, optional (default=1.0)
        Called theta in the Leiden algorithm, this is used to scale
        modularity gain in Leiden refinement phase, to compute
        the probability of joining a random leiden community.

    Returns
    -------
    parts : cudf.DataFrame
        GPU data frame of size V containing two columns the vertex id and the
        partition id it is assigned to.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['partition'] : cudf.Series
            Contains the partition assigned to the vertices

    modularity_score : float
        a floating point number containing the global modularity score of the
        partitioning.

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> parts, modularity_score = hipgraph.leiden(G)

    """
    G, isNx = ensure_hipgraph_obj_for_nx(G)

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    vertex, partition, modularity_score = pylibhipgraph_leiden(
        resource_handle=ResourceHandle(),
        random_state=random_state,
        graph=G._plc_graph,
        max_level=max_iter,
        resolution=resolution,
        theta=theta,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["partition"] = partition

    if G.renumbered:
        parts = G.unrenumber(df, "vertex")
    else:
        parts = df

    if isNx is True:
        parts = df_score_to_dictionary(df, "partition")

    return parts, modularity_score
