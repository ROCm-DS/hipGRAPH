# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph_functions cimport (
    hipgraph_allgather,
    hipgraph_induced_subgraph_get_destinations,
    hipgraph_induced_subgraph_get_edge_ids,
    hipgraph_induced_subgraph_get_edge_type_ids,
    hipgraph_induced_subgraph_get_edge_weights,
    hipgraph_induced_subgraph_get_sources,
    hipgraph_induced_subgraph_get_subgraph_offsets,
    hipgraph_induced_subgraph_result_free,
    hipgraph_induced_subgraph_result_t,
)
from pylibhipgraph._hipgraph_c.resource_handle cimport hipgraph_resource_handle_t
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport (
    assert_CAI_type,
    assert_success,
    copy_to_cupy_array,
    create_hipgraph_type_erased_device_array_view_from_py_obj,
)


def replicate_edgelist(ResourceHandle resource_handle,
                       src_array,
                       dst_array,
                       weight_array,
                       edge_id_array,
                       edge_type_id_array):
    """
        Replicate edges across all GPUs

        Parameters
        ----------
        resource_handle : ResourceHandle
            Handle to the underlying device resources needed for referencing data
            and running algorithms.

        src_array : device array type, optional
            Device array containing the vertex identifiers of the source of each
            directed edge. The order of the array corresponds to the ordering of the
            dst_array, where the ith item in src_array and the ith item in dst_array
            define the ith edge of the graph.

        dst_array : device array type, optional
            Device array containing the vertex identifiers of the destination of
            each directed edge. The order of the array corresponds to the ordering
            of the src_array, where the ith item in src_array and the ith item in
            dst_array define the ith edge of the graph.

        weight_array : device array type, optional
            Device array containing the weight values of each directed edge. The
            order of the array corresponds to the ordering of the src_array and
            dst_array arrays, where the ith item in weight_array is the weight value
            of the ith edge of the graph.

        edge_id_array : device array type, optional
            Device array containing the edge id values of each directed edge. The
            order of the array corresponds to the ordering of the src_array and
            dst_array arrays, where the ith item in edge_id_array is the id value
            of the ith edge of the graph.

        edge_type_id_array : device array type, optional
            Device array containing the edge type id values of each directed edge. The
            order of the array corresponds to the ordering of the src_array and
            dst_array arrays, where the ith item in edge_type_id_array is the type id
            value of the ith edge of the graph.

        Returns
        -------
        return cupy arrays of 'src' and/or 'dst' and/or 'weight'and/or 'edge_id'
        and/or 'edge_type_id'.
    """
    assert_CAI_type(src_array, "src_array", True)
    assert_CAI_type(dst_array, "dst_array", True)
    assert_CAI_type(weight_array, "weight_array", True)
    assert_CAI_type(edge_id_array, "edge_id_array", True)
    assert_CAI_type(edge_type_id_array, "edge_type_id_array", True)
    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr

    cdef hipgraph_induced_subgraph_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cdef hipgraph_type_erased_device_array_view_t* srcs_view_ptr = \
        create_hipgraph_type_erased_device_array_view_from_py_obj(src_array)

    cdef hipgraph_type_erased_device_array_view_t* dsts_view_ptr = \
        create_hipgraph_type_erased_device_array_view_from_py_obj(dst_array)


    cdef hipgraph_type_erased_device_array_view_t* weights_view_ptr = \
        create_hipgraph_type_erased_device_array_view_from_py_obj(weight_array)

    cdef hipgraph_type_erased_device_array_view_t* edge_ids_view_ptr = \
        create_hipgraph_type_erased_device_array_view_from_py_obj(edge_id_array)

    cdef hipgraph_type_erased_device_array_view_t* edge_type_ids_view_ptr = \
        create_hipgraph_type_erased_device_array_view_from_py_obj(edge_type_id_array)

    error_code = hipgraph_allgather(c_resource_handle_ptr,
                                   srcs_view_ptr,
                                   dsts_view_ptr,
                                   weights_view_ptr,
                                   edge_ids_view_ptr,
                                   edge_type_ids_view_ptr,
                                   &result_ptr,
                                   &error_ptr)
    assert_success(error_code, error_ptr, "replicate_edgelist")
    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* sources_ptr
    if src_array is not None:
        sources_ptr = hipgraph_induced_subgraph_get_sources(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* destinations_ptr
    if dst_array is not None:
        destinations_ptr = hipgraph_induced_subgraph_get_destinations(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* edge_weights_ptr = \
        hipgraph_induced_subgraph_get_edge_weights(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* edge_ids_ptr = \
        hipgraph_induced_subgraph_get_edge_ids(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* edge_type_ids_ptr = \
        hipgraph_induced_subgraph_get_edge_type_ids(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* subgraph_offsets_ptr = \
        hipgraph_induced_subgraph_get_subgraph_offsets(result_ptr)

    # FIXME: Get ownership of the result data instead of performing a copy
    # for perfomance improvement

    cupy_sources = None
    cupy_destinations = None
    cupy_edge_weights = None
    cupy_edge_ids = None
    cupy_edge_type_ids = None

    if src_array is not None:
        cupy_sources = copy_to_cupy_array(
            c_resource_handle_ptr, sources_ptr)

    if dst_array is not None:
        cupy_destinations = copy_to_cupy_array(
            c_resource_handle_ptr, destinations_ptr)

    if weight_array is not None:
        cupy_edge_weights = copy_to_cupy_array(
            c_resource_handle_ptr, edge_weights_ptr)

    if edge_id_array is not None:
        cupy_edge_ids = copy_to_cupy_array(
            c_resource_handle_ptr, edge_ids_ptr)

    if edge_type_id_array is not None:
        cupy_edge_type_ids = copy_to_cupy_array(
            c_resource_handle_ptr, edge_type_ids_ptr)

    cupy_subgraph_offsets = copy_to_cupy_array(
        c_resource_handle_ptr, subgraph_offsets_ptr)

    # Free pointer
    hipgraph_induced_subgraph_result_free(result_ptr)
    if src_array is not None:
        hipgraph_type_erased_device_array_view_free(srcs_view_ptr)
    if dst_array is not None:
        hipgraph_type_erased_device_array_view_free(dsts_view_ptr)
    if weight_array is not None:
        hipgraph_type_erased_device_array_view_free(weights_view_ptr)
    if edge_id_array is not None:
        hipgraph_type_erased_device_array_view_free(edge_ids_view_ptr)
    if edge_type_id_array is not None:
        hipgraph_type_erased_device_array_view_free(edge_type_ids_view_ptr)

    return (cupy_sources, cupy_destinations,
            cupy_edge_weights, cupy_edge_ids,
            cupy_edge_type_ids, cupy_subgraph_offsets)
