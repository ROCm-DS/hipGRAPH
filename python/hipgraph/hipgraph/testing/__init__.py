# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from hipgraph.datasets import (  # twitter,
    amazon0302,
    cit_patents,
    cyber,
    dining_prefs,
    dolphins,
    email_Eu_core,
    europe_osm,
    hollywood,
    karate,
    karate_disjoint,
    netscience,
    polbooks,
    small_line,
    small_tree,
    soc_livejournal,
    toy_graph,
    toy_graph_undirected,
)
from hipgraph.testing.resultset import (
    Resultset,
    default_resultset_download_dir,
    get_resultset,
    load_resultset,
)
from hipgraph.testing.utils import RAPIDS_DATASET_ROOT_DIR, RAPIDS_DATASET_ROOT_DIR_PATH

#
# Moved Dataset Batches
#

UNDIRECTED_DATASETS = [karate, dolphins]
SMALL_DATASETS = [karate, dolphins, polbooks]
WEIGHTED_DATASETS = [
    dining_prefs,
    dolphins,
    karate,
    karate_disjoint,
    netscience,
    polbooks,
    small_line,
    small_tree,
]
ALL_DATASETS = [
    dining_prefs,
    dolphins,
    karate,
    karate_disjoint,
    polbooks,
    netscience,
    small_line,
    small_tree,
    email_Eu_core,
    toy_graph,
    toy_graph_undirected,
]
DEFAULT_DATASETS = [dolphins, netscience, karate_disjoint]
BENCHMARKING_DATASETS = [
    soc_livejournal,
    cit_patents,
    europe_osm,
    hollywood,
    amazon0302,
]
