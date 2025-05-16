.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.HipGraphRandomState, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-HipGraphRandomState:

*******************************************
pylibhipgraph.HipGraphRandomState
*******************************************

**class HipGraphRandomState**

This class wraps a ``hipgraph_rng_state_t`` instance, which represents a
random state.

__cinit__ (*self, ResourceHandle resource_handle, seed=None*)

Constructs a new HipGraphRandomState instance.

Parameters
----------

resource_handle: pylibhipgraph.ResourceHandle (Required)
    The hipgraph resource handle for this process.
seed: int (Optional)
    The random seed of this random state object.
    Defaults to the hash of the hostname, pid, and time.

__dealloc__(*self*)

Destroys this HipGraphRandomState instance.  Properly calls
free to destroy the underlying C++ object.
