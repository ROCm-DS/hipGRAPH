# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# ######################################################################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ######################################################################################################################

cmake_policy(PUSH)
cmake_policy(SET CMP0076 NEW)

set(HIPGRAPH_C_TEST_SOURCES
    betweenness_centrality_test.cpp
    bfs_test.cpp
    core_number_test.cpp
    create_graph_test.cpp
    degrees_test.cpp
    ecg_test.cpp
    edge_betweenness_centrality_test.cpp
    egonet_test.cpp
    eigenvector_centrality_test.cpp
    extract_paths_test.cpp
    generate_rmat_test.cpp
    hits_test.cpp
    induced_subgraph_test.cpp
    katz_test.cpp
    k_core_test.cpp
    leiden_test.cpp
    louvain_test.cpp
    # These need mg_test_utils*, MPI, etc. mg_betweenness_centrality_test.cpp mg_bfs_test.cpp mg_core_number_test.cpp
    # mg_create_graph_test.cpp mg_degrees_test.cpp mg_ecg_test.cpp mg_edge_betweenness_centrality_test.cpp
    # mg_egonet_test.cpp mg_eigenvector_centrality_test.cpp mg_generate_rmat_test.cpp mg_hits_test.cpp
    # mg_induced_subgraph_test.cpp mg_katz_test.cpp mg_k_core_test.cpp mg_leiden_test.cpp mg_louvain_test.cpp
    # mg_pagerank_test.cpp mg_random_walks_test.cpp mg_similarity_test.cpp mg_sssp_test.cpp
    # mg_strongly_connected_components_test.cpp mg_triangle_count_test.cpp mg_two_hop_neighbors_test.cpp
    # mg_uniform_neighbor_sample_test.cpp mg_weakly_connected_components_test.cpp Crashes: node2vec_test.cpp
    pagerank_test.cpp
    sg_random_walks_test.cpp
    similarity_test.cpp
    sssp_test.cpp
    strongly_connected_components_test.cpp
    triangle_count_test.cpp
    two_hop_neighbors_test.cpp
    uniform_neighbor_sample_test.cpp
    weakly_connected_components_test.cpp)

target_sources(hipgraph-test PRIVATE ${HIPGRAPH_C_TEST_SOURCES} test_utils.cpp)

cmake_policy(POP)
