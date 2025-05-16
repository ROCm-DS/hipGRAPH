.. meta::
  :description: hipGRAPH documentation and API reference library
  :keywords: Graph, Graph-algorithms, Graph-analysis, Graph-processing, Complex-networks, rocGraph, hipGraph, cuGraph, NetworkX, GPU, RAPIDS, ROCm-DS

.. _hipgraph_community_functions_:

********************************************************************
Community Functions
********************************************************************

Triangle Counting
-----------------
.. doxygenfunction:: hipgraph_triangle_count

Louvain
-------
.. doxygenfunction:: hipgraph_louvain

Leiden
------
.. doxygenfunction:: hipgraph_leiden

ECG
---
.. doxygenfunction:: hipgraph_ecg

Extract Egonet
--------------
.. doxygenfunction:: hipgraph_extract_ego

Balanced Cut
------------
.. doxygenfunction:: hipgraph_balanced_cut_clustering

Spectral Clustering - Modularity Maximization
---------------------------------------------
.. doxygenfunction:: hipgraph_spectral_modularity_maximization

.. doxygenfunction:: hipgraph_analyze_clustering_modularity

Spectral Clustering - Edge Cut
------------------------------
.. doxygenfunction:: hipgraph_analyze_clustering_edge_cut

.. doxygenfunction:: hipgraph_analyze_clustering_ratio_cut


Community Support Functions
---------------------------
 .. doxygengroup:: community
     :members:
     :content-only:
