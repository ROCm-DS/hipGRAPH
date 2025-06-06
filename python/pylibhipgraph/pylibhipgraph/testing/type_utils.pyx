# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t

import cupy

from pylibhipgraph._hipgraph_c.algorithms cimport (
    hipgraph_sample_result_free,
    hipgraph_sample_result_t,
)
from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_create,
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.resource_handle cimport hipgraph_resource_handle_t
from pylibhipgraph._hipgraph_c.sampling_algorithms cimport (
    hipgraph_test_uniform_neighborhood_sample_result_create,
)
from pylibhipgraph.internal_types.sampling_result cimport SamplingResult
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport (
    assert_CAI_type,
    assert_success,
    get_c_type_from_numpy_type,
)


def create_sampling_result(ResourceHandle resource_handle,
		           device_sources,
                           device_destinations,
                           device_weights,
                           device_edge_id,
                           device_edge_type,
                           device_hop,
                           device_batch_label):
    """
    Create a SamplingResult object from individual host arrays.

    This function is currently testing-only because the SamplingResult type is
    considered internal (ie. pylibhipgraph users will not be exposed to it) and
    because SamplingResult instances will be created from a
    hipgraph_sample_result_t pointer and not host arrays.
    """
    assert_CAI_type(device_sources, "device_sources")
    assert_CAI_type(device_destinations, "device_destinations")
    if device_weights is not None:
        assert_CAI_type(device_weights, "device_weights")
    if device_edge_id is not None:
        assert_CAI_type(device_edge_id, "device_edge_id")
    if device_edge_type is not None:
        assert_CAI_type(device_edge_type, "device_edge_type")
    if device_weights is not None:
        assert_CAI_type(device_weights, "device_weights")
    if device_hop is not None:
        assert_CAI_type(device_hop, "device_hop")
    if device_batch_label is not None:
        assert_CAI_type(device_batch_label, "device_batch_label")

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr

    cdef hipgraph_sample_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cdef uintptr_t cai_srcs_ptr = \
        device_sources.__cuda_array_interface__["data"][0]
    cdef uintptr_t cai_dsts_ptr = \
        device_destinations.__cuda_array_interface__["data"][0]

    cdef uintptr_t cai_weights_ptr
    if device_weights is not None:
        cai_weights_ptr = device_weights.__cuda_array_interface__['data'][0]
    cdef uintptr_t cai_edge_ids_ptr
    if device_edge_id is not None:
        cai_edge_ids_ptr = device_edge_id.__cuda_array_interface__['data'][0]
    cdef uintptr_t cai_edge_types_ptr
    if device_edge_type is not None:
        cai_edge_types_ptr = device_edge_type.__cuda_array_interface__['data'][0]
    cdef uintptr_t cai_hop_ptr
    if device_hop is not None:
        cai_hop_ptr = device_hop.__cuda_array_interface__['data'][0]
    cdef uintptr_t cai_batch_id_ptr
    if device_batch_label is not None:
        cai_batch_id_ptr = device_batch_label.__cuda_array_interface__['data'][0]

    cdef hipgraph_type_erased_device_array_view_t* c_srcs_view_ptr = (
        hipgraph_type_erased_device_array_view_create(
            <void*>cai_srcs_ptr,
            len(device_sources),
            get_c_type_from_numpy_type(device_sources.dtype))
    )
    cdef hipgraph_type_erased_device_array_view_t* c_dsts_view_ptr = (
        hipgraph_type_erased_device_array_view_create(
            <void*>cai_dsts_ptr,
            len(device_destinations),
            get_c_type_from_numpy_type(device_destinations.dtype))
    )
    cdef hipgraph_type_erased_device_array_view_t* c_weight_ptr = <hipgraph_type_erased_device_array_view_t*>NULL
    if device_weights is not None:
        c_weight_ptr = (
            hipgraph_type_erased_device_array_view_create(
                <void*>cai_weights_ptr,
                len(device_weights),
                get_c_type_from_numpy_type(device_weights.dtype)
            )
        )
    cdef hipgraph_type_erased_device_array_view_t* c_edge_id_ptr = <hipgraph_type_erased_device_array_view_t*>NULL
    if device_weights is not None:
        c_edge_id_ptr = (
            hipgraph_type_erased_device_array_view_create(
                <void*>cai_edge_ids_ptr,
                len(device_edge_id),
                get_c_type_from_numpy_type(device_edge_id.dtype)
            )
        )
    cdef hipgraph_type_erased_device_array_view_t* c_edge_type_ptr = <hipgraph_type_erased_device_array_view_t*>NULL
    if device_weights is not None:
        c_edge_type_ptr = (
            hipgraph_type_erased_device_array_view_create(
                <void*>cai_edge_types_ptr,
                len(device_edge_type),
                get_c_type_from_numpy_type(device_edge_type.dtype)
            )
        )

    cdef hipgraph_type_erased_device_array_view_t* c_hop_ptr = <hipgraph_type_erased_device_array_view_t*>NULL
    if device_weights is not None:
        c_hop_ptr = (
            hipgraph_type_erased_device_array_view_create(
                <void*>cai_hop_ptr,
                len(device_hop),
                get_c_type_from_numpy_type(device_hop.dtype)
            )
        )

    cdef hipgraph_type_erased_device_array_view_t* c_label_ptr = <hipgraph_type_erased_device_array_view_t*>NULL
    if device_weights is not None:
        c_label_ptr = (
            hipgraph_type_erased_device_array_view_create(
                <void*>cai_batch_id_ptr,
                len(device_batch_label),
                get_c_type_from_numpy_type(device_batch_label.dtype)
            )
        )


    error_code = hipgraph_test_uniform_neighborhood_sample_result_create(
        c_resource_handle_ptr,
        c_srcs_view_ptr,
        c_dsts_view_ptr,
        c_edge_id_ptr,
        c_edge_type_ptr,
        c_weight_ptr,
        c_hop_ptr,
        c_label_ptr,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "create_sampling_result")

    result = SamplingResult()
    result.set_ptr(result_ptr)

    # Free the non-owning view containers. This should not free result data.
    hipgraph_type_erased_device_array_view_free(c_srcs_view_ptr)
    hipgraph_type_erased_device_array_view_free(c_dsts_view_ptr)
    hipgraph_type_erased_device_array_view_free(c_edge_id_ptr)
    hipgraph_type_erased_device_array_view_free(c_edge_type_ptr)
    hipgraph_type_erased_device_array_view_free(c_weight_ptr)
    hipgraph_type_erased_device_array_view_free(c_hop_ptr)
    hipgraph_type_erased_device_array_view_free(c_label_ptr)

    return result
