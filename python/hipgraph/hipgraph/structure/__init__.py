# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from hipgraph.structure.convert_matrix import (
    from_adjlist,
    from_cudf_edgelist,
    from_edgelist,
    from_numpy_array,
    from_numpy_matrix,
    from_pandas_adjacency,
    from_pandas_edgelist,
    to_numpy_array,
    to_numpy_matrix,
    to_pandas_adjacency,
    to_pandas_edgelist,
)
from hipgraph.structure.graph_classes import (
    BiPartiteGraph,
    Graph,
    MultiGraph,
    is_bipartite,
    is_directed,
    is_multigraph,
    is_multipartite,
    is_weighted,
)
from hipgraph.structure.hypergraph import hypergraph
from hipgraph.structure.number_map import NumberMap
from hipgraph.structure.replicate_edgelist import (
    replicate_cudf_dataframe,
    replicate_cudf_series,
    replicate_edgelist,
)
from hipgraph.structure.symmetrize import symmetrize, symmetrize_ddf, symmetrize_df
