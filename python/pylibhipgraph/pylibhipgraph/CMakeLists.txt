# Copyright (c) 2022-2024, NVIDIA CORPORATION. SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# add_subdirectory(components) # Needs updated to the non-legacy components algorithms, but isn't used in NetworkX
add_subdirectory(internal_types)
add_subdirectory(testing)

set(cython_sources
    analyze_clustering_edge_cut.pyx
    analyze_clustering_modularity.pyx
    analyze_clustering_ratio_cut.pyx
    balanced_cut_clustering.pyx
    betweenness_centrality.pyx
    bfs.pyx
    core_number.pyx
    ecg.pyx
    edge_betweenness_centrality.pyx
    egonet.pyx
    eigenvector_centrality.pyx
    generate_rmat_edgelist.pyx
    generate_rmat_edgelists.pyx
    graph_properties.pyx
    graphs.pyx
    hits.pyx
    induced_subgraph.pyx
    k_core.pyx
    k_truss_subgraph.pyx
    jaccard_coefficients.pyx
    sorensen_coefficients.pyx
    overlap_coefficients.pyx
    # cosine_coefficients.pyx # Needs hipgraph_ops
    katz_centrality.pyx
    leiden.pyx
    louvain.pyx
    node2vec.pyx
    pagerank.pyx
    personalized_pagerank.pyx
    random.pyx
    resource_handle.pyx
    spectral_modularity_maximization.pyx
    select_random_vertices.pyx
    sssp.pyx
    triangle_count.pyx
    two_hop_neighbors.pyx
    # uniform_neighbor_sample.pyx # Needs hipgraph_ops
    uniform_random_walks.pyx
    utils.pyx
    weakly_connected_components.pyx
    replicate_edgelist.pyx
    degrees.pyx
    all_pairs_jaccard_coefficients.pyx
    all_pairs_sorensen_coefficients.pyx
    all_pairs_overlap_coefficients.pyx
    # all_pairs_cosine_coefficients.pyx # Needs hipgraph_ops
)
set(linked_libraries roc::hipgraph)

rapids_cython_create_modules(CXX SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES ${linked_libraries}
                             ASSOCIATED_TARGETS hipgraph)
