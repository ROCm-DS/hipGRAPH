.. meta::
  :description: hipGRAPH documentation and API reference library
  :keywords: Graph, Graph-algorithms, Graph-analysis, Graph-processing, Complex-networks, rocGraph, hipGraph, cuGraph, NetworkX, GPU, RAPIDS, ROCm-DS

.. _hipgraph-python:

*********************************
hipGRAPH Python API documentation
*********************************

.. note::
  The early access version of the hipGRAPH Python layer is untested and the Python API is unsupported.

This directory contains the sources to the `pylibhipgraph` package. The sources
are primarily cython files which are built using the `setup.py` file in the
parent directory and depend on the `libhipgraph_c` and `libhipgraph` libraries and
headers.

.. toctree::
   :maxdepth: 1

   pyx_files/all_pairs_cosine_coefficients.rst
   pyx_files/all_pairs_jaccard_coefficients.rst
   pyx_files/all_pairs_overlap_coefficients.rst
   pyx_files/all_pairs_sorensen_coefficients.rst
   pyx_files/analyze_clustering_edge_cut.rst
   pyx_files/analyze_clustering_modularity.rst
   pyx_files/analyze_clustering_ratio_cut.rst
   pyx_files/balanced_cut_clustering.rst
   pyx_files/betweenness_centrality.rst
   pyx_files/bfs.rst
   pyx_files/core_number.rst
   pyx_files/cosine_coefficients.rst
   pyx_files/count_multi_edges.rst
   pyx_files/degrees.rst
   pyx_files/ecg.rst
   pyx_files/edge_betweenness_centrality.rst
   pyx_files/egonet.rst
   pyx_files/eigenvector_centrality.rst
   pyx_files/generate_rmat_edgelist.rst
   pyx_files/generate_rmat_edgelists.rst
   pyx_files/graph_properties.rst
   pyx_files/graphs.rst
   pyx_files/hits.rst
   pyx_files/induced_subgraph.rst
   pyx_files/jaccard_coefficients.rst
   pyx_files/k_core.rst
   pyx_files/k_truss_subgraph.rst
   pyx_files/katz_centrality.rst
   pyx_files/leiden.rst
   pyx_files/louvain.rst
   pyx_files/node2vec.rst
   pyx_files/overlap_coefficients.rst
   pyx_files/pagerank.rst
   pyx_files/personalized_pagerank.rst
   pyx_files/random.rst
   pyx_files/replicate_edgelist.rst
   pyx_files/resource_handle.rst
   pyx_files/select_random_vertices.rst
   pyx_files/sorensen_coefficients.rst
   pyx_files/spectral_modularity_maximization.rst
   pyx_files/sssp.rst
   pyx_files/triangle_count.rst
   pyx_files/two_hop_neighbors.rst
   pyx_files/uniform_neighbor_sample.rst
   pyx_files/uniform_random_walks.rst
   pyx_files/utils.rst
   pyx_files/weakly_connected_components.rst
