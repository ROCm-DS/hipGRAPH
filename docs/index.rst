.. meta::
  :description: hipGRAPH documentation and API reference library
  :keywords: Graph, Graph-algorithms, Graph-analysis, Graph-processing, Complex-networks, rocGraph, hipGraph, cuGraph, NetworkX, GPU, RAPIDS, ROCm-DS

.. _hipgraph:

********************************************************************
hipGRAPH documentation
********************************************************************

.. note::
  hipGRAPH is in an early access state. Running production workloads is not recommended. The early access version of the hipGRAPH Python layer is untested and the Python API is unsupported.

hipGRAPH is a graph marshalling library that acts as a wrapper between your application and a worker graph library, such as `rocGRAPH <https://rocm.docs.amd.com/projects/rocGRAPH/>`_.

The hipGRAPH library contains a collection of graph algorithms, enabling you to build, analyze, and manipulate complex graphs or networks. hipGRAPH is derived from the cuGraph library that forms part of the NVIDIA RAPIDS open source project. The hipGRAPH library described here should not be confused with `HIP graphs <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html>`_ in the HIP runtime API.

The hipGRAPH code is open and hosted at `https://github.com/ROCm-DS/hipGraph <https://github.com/ROCm-DS/hipGraph>`_.

The hipGRAPH documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :ref:`linux-install`

  .. grid-item-card:: API reference

    * :ref:`C++ API reference <hipgraph-reference>`
    * :ref:`Python API reference <hipgraph-python>`

To contribute to the documentation refer to `Contributing to ROCm-DS  <https://rocm.docs.amd.com/projects/ROCm-DS/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/projects/ROCm-DS/latest/about/license.html>`_ page.
