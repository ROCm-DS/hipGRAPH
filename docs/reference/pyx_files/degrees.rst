.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.degrees, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-degrees:

*******************************************
pylibhipgraph.degrees
*******************************************

pylibhipgraph.in_degrees
=========================

**in_degrees** (*ResourceHandle resource_handle, _GPUGraph graph, source_vertices, bool_t do_expensive_check*)

Compute the in degrees for the nodes of the graph.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

source_vertices : cupy array
    The nodes for which we will compute degrees.

do_expensive_check : bool_t
    A flag to run expensive checks for input arguments if True.

Returns
-------

A tuple of device arrays, where the first item in the tuple is a device
array containing the vertices, the second item in the tuple is a device
array containing the in degrees for the vertices.

Examples
--------

.. code:: Python

    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, in_degrees) = pylibhipgraph.in_degrees(
                                   resource_handle, G, None, False)

pylibhipgraph.out_degrees
=========================

**out_degrees** (*ResourceHandle resource_handle _GPUGraph graph, source_vertices, bool_t do_expensive_check*)

Compute the out degrees for the nodes of the graph.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

source_vertices : cupy array
    The nodes for which we will compute degrees.

do_expensive_check : bool_t
    A flag to run expensive checks for input arguments if True.

Returns
-------

A tuple of device arrays, where the first item in the tuple is a device
array containing the vertices, the second item in the tuple is a device
array containing the out degrees for the vertices.

Examples
--------

.. code:: Python

    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, out_degrees) = pylibhipgraph.out_degrees(
                                    resource_handle, G, None, False)


pylibhipgraph.degrees
=========================

**degrees** (*ResourceHandle resource_handle, _GPUGraph graph, source_vertices, bool_t do_expensive_check*)

Compute the degrees for the nodes of the graph.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

source_vertices : cupy array
    The nodes for which we will compute degrees.

do_expensive_check : bool_t
    A flag to run expensive checks for input arguments if True.

Returns
-------

A tuple of device arrays, where the first item in the tuple is a device
array containing the vertices, the second item in the tuple is a device
array containing the in degrees for the vertices, the third item in the
tuple is a device array containing the out degrees for the vertices.

Examples
--------

.. code:: Python

    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, in_degrees, out_degrees) = pylibhipgraph.degrees(
                                                resource_handle, G, None, False)
