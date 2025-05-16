.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.copy_to_cupy_array, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-copy_to_cupy_array:

*******************************************
pylibhipgraph.copy_to_cupy_array
*******************************************

**copy_to_cupy_array** (hipgraph_resource_handle_t\* c_resource_handle_ptr, hipgraph_type_erased_device_array_view_t\* device_array_view_ptr)

Copy the contents from a device array view as returned by various hipgraph_*
APIs to a new cupy device array, typically intended to be used as a return
value from pylibhipgraph APIs.

**copy_to_cupy_array_ids** (hipgraph_resource_handle_t\* c_resource_handle_ptr, hipgraph_type_erased_device_array_view_t\* device_array_view_ptr)

Copy the contents from a device array view as returned by various hipgraph_*
APIs to a new cupy device array, typically intended to be used as a return
value from pylibhipgraph APIs then convert float to int
