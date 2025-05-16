# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from pathlib import Path

from hipgraph.datasets import metadata

# datasets module
from hipgraph.datasets.dataset import (
    Dataset,
    default_download_dir,
    download_all,
    get_download_dir,
    set_download_dir,
)

# metadata path for .yaml files
meta_path = Path(__file__).parent / "metadata"

cyber = Dataset(meta_path / "cyber.yaml")
dining_prefs = Dataset(meta_path / "dining_prefs.yaml")
dolphins = Dataset(meta_path / "dolphins.yaml")
email_Eu_core = Dataset(meta_path / "email_Eu_core.yaml")
karate = Dataset(meta_path / "karate.yaml")
karate_asymmetric = Dataset(meta_path / "karate_asymmetric.yaml")
karate_disjoint = Dataset(meta_path / "karate_disjoint.yaml")
netscience = Dataset(meta_path / "netscience.yaml")
polbooks = Dataset(meta_path / "polbooks.yaml")
small_line = Dataset(meta_path / "small_line.yaml")
small_tree = Dataset(meta_path / "small_tree.yaml")
toy_graph = Dataset(meta_path / "toy_graph.yaml")
toy_graph_undirected = Dataset(meta_path / "toy_graph_undirected.yaml")

# Benchmarking datasets: be mindful of memory usage
# 250 MB
soc_livejournal = Dataset(meta_path / "soc-livejournal1.yaml")
# 965 MB
cit_patents = Dataset(meta_path / "cit-patents.yaml")
# 1.8 GB
europe_osm = Dataset(meta_path / "europe_osm.yaml")
# 1.5 GB
hollywood = Dataset(meta_path / "hollywood.yaml")
amazon0302 = Dataset(meta_path / "amazon0302.yaml")
