.. meta::
  :description: hipGRAPH documentation and API reference library
  :keywords: Graph, Graph-algorithms, Graph-analysis, Graph-processing, Complex-networks, rocGraph, hipGraph, cuGraph, NetworkX, GPU, RAPIDS, ROCm-DS

.. _hipgraph_traversal_functions_:

********************************************************************
Traversal Functions
********************************************************************


Breadth First Search (BFS)
--------------------------
.. doxygenfunction:: hipgraph_bfs

Single-Source Shortest-Path (SSSP)
----------------------------------
.. doxygenfunction:: hipgraph_sssp

Path Extraction
---------------
.. doxygenfunction:: hipgraph_extract_paths

Extract Max Path Length
-----------------------
.. doxygenfunction:: hipgraph_extract_paths_result_get_max_path_length

Traversal Support Functions
---------------------------
.. doxygengroup:: traversal
     :members:
     :content-only:
