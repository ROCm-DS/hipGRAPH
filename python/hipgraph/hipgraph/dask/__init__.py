# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from dask import config

from .centrality.betweenness_centrality import (
    betweenness_centrality,
    edge_betweenness_centrality,
)
from .centrality.eigenvector_centrality import eigenvector_centrality
from .centrality.katz_centrality import katz_centrality
from .common.read_utils import get_chunksize, get_n_workers
from .community.egonet import ego_graph
from .community.induced_subgraph import induced_subgraph
from .community.ktruss_subgraph import ktruss_subgraph
from .community.leiden import leiden
from .community.louvain import louvain
from .community.triangle_count import triangle_count
from .components.connectivity import weakly_connected_components
from .cores.core_number import core_number
from .cores.k_core import k_core
from .link_analysis.hits import hits
from .link_analysis.pagerank import pagerank
from .link_prediction.cosine import all_pairs_cosine, cosine
from .link_prediction.jaccard import all_pairs_jaccard, jaccard
from .link_prediction.overlap import all_pairs_overlap, overlap
from .link_prediction.sorensen import all_pairs_sorensen, sorensen
from .sampling.random_walks import random_walks
from .sampling.uniform_neighbor_sample import uniform_neighbor_sample
from .traversal.bfs import bfs
from .traversal.sssp import sssp

# Avoid "p2p" shuffling in dask for now
config.set({"dataframe.shuffle.method": "tasks"})
