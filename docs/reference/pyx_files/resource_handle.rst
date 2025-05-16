.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.ResourceHandle, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-ResourceHandle:

*******************************************
pylibhipgraph.ResourceHandle
*******************************************

**class ResourceHandle**

RAII-stye resource handle class to manage individual create/free calls and
the corresponding pointer to a hipgraph_resource_handle_t

__cinit__(self, handle=None)

cdef void* handle_ptr = NULL
cdef size_t handle_size_t
if handle is not None:

  # FIXME: rather than assume a RAFT handle here, consider something
  # like a factory function in hipgraph (which already has a RAFT
  # dependency and makes RAFT assumptions) that takes a RAFT handle
  # and constructs/returns a ResourceHandle
  handle_size_t = <size_t>handle
  handle_ptr = <void \*> handle_size_t

self.c_resource_handle_ptr[0] = hipgraph_create_resource_handle(handle_ptr)[0]
# FIXME: check for error

def __dealloc__(self)

# FIXME: free only if handle is a valid pointer
hipgraph_free_resource_handle(self.c_resource_handle_ptr)
