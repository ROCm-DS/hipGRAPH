# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import warnings
from typing import Union

import cudf
import hipgraph
from hipgraph.structure import Graph
from hipgraph.utilities.utils import import_optional

# FIXME: the networkx.Graph type used in the type annotation for subgraph() is
# specified using a string literal to avoid depending on and importing
# networkx. Instead, networkx is imported optionally, which may cause a problem
# for a type checker if run in an environment where networkx is not installed.
networkx = import_optional("networkx")


def subgraph(
    G: Union[Graph, "networkx.Graph"],
    vertices: Union[cudf.Series, cudf.DataFrame],
) -> Union[Graph, "networkx.Graph"]:
    """
    Compute a subgraph of the existing graph including only the specified
    vertices.  This algorithm works with both directed and undirected graphs
    and does not actually traverse the edges, but instead simply pulls out any
    edges that are incident on vertices that are both contained in the vertices
    list.

    If no subgraph can be extracted from the vertices provided, a 'None' value
    will be returned.

    Parameters
    ----------
    G : hipgraph.Graph or networkx.Graph
        The current implementation only supports weighted graphs.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    vertices : cudf.Series or cudf.DataFrame
        Specifies the vertices of the induced subgraph. For multi-column
        vertices, vertices should be provided as a cudf.DataFrame

    Returns
    -------
    Sg : hipgraph.Graph or networkx.Graph
        A graph object containing the subgraph induced by the given vertex set.

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> verts = np.zeros(3, dtype=np.int32)
    >>> verts[0] = 0
    >>> verts[1] = 1
    >>> verts[2] = 2
    >>> sverts = cudf.Series(verts)
    >>> Sg = hipgraph.subgraph(G, sverts)  # doctest: +SKIP
    """

    warning_msg = (
        "This call is deprecated. Please call 'hipgraph.induced_subgraph()' instead."
    )
    warnings.warn(warning_msg, DeprecationWarning)

    result_graph, _ = hipgraph.induced_subgraph(G, vertices)

    return result_graph
