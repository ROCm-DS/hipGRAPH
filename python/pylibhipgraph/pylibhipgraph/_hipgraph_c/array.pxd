# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    byte_t,
    data_type_id_t,
    hipgraph_resource_handle_t,
)


cdef extern from "hipgraph/hipgraph_c/array.h":

    ctypedef struct hipgraph_type_erased_device_array_t:
        pass

    ctypedef struct hipgraph_type_erased_device_array_view_t:
        pass

    ctypedef struct hipgraph_type_erased_host_array_t:
        pass

    ctypedef struct hipgraph_type_erased_host_array_view_t:
        pass

    cdef hipgraph_error_code_t \
        hipgraph_type_erased_device_array_create(
            const hipgraph_resource_handle_t* handle,
            data_type_id_t dtype,
            size_t n_elems,
            hipgraph_type_erased_device_array_t** array,
            hipgraph_error_t** error
        )

    cdef void \
        hipgraph_type_erased_device_array_free(
            hipgraph_type_erased_device_array_t* p
        )

    # cdef void* \
    #     hipgraph_type_erased_device_array_release(
    #         hipgraph_type_erased_device_array_t* p
    #     )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_type_erased_device_array_view(
            hipgraph_type_erased_device_array_t* array
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_type_erased_device_array_view_create(
            void* pointer,
            size_t n_elems,
            data_type_id_t dtype
        )

    cdef void \
        hipgraph_type_erased_device_array_view_free(
            hipgraph_type_erased_device_array_view_t* p
        )

    cdef size_t \
        hipgraph_type_erased_device_array_view_size(
            const hipgraph_type_erased_device_array_view_t* p
        )

    cdef data_type_id_t \
        hipgraph_type_erased_device_array_view_type(
            const hipgraph_type_erased_device_array_view_t* p
        )

    cdef const void* \
        hipgraph_type_erased_device_array_view_pointer(
            const hipgraph_type_erased_device_array_view_t* p
        )

    cdef hipgraph_error_code_t \
        hipgraph_type_erased_host_array_create(
            const hipgraph_resource_handle_t* handle,
            data_type_id_t dtype,
            size_t n_elems,
            hipgraph_type_erased_host_array_t** array,
            hipgraph_error_t** error
        )

    cdef void \
        hipgraph_type_erased_host_array_free(
            hipgraph_type_erased_host_array_t* p
        )

    # cdef void* \
    #     hipgraph_type_erased_host_array_release(
    #         hipgraph_type_erased_host_array_t* p
    #     )

    cdef hipgraph_type_erased_host_array_view_t* \
        hipgraph_type_erased_host_array_view(
            hipgraph_type_erased_host_array_t* array
        )

    cdef hipgraph_type_erased_host_array_view_t* \
        hipgraph_type_erased_host_array_view_create(
            void* pointer,
            size_t n_elems,
            data_type_id_t dtype
        )

    cdef void \
        hipgraph_type_erased_host_array_view_free(
            hipgraph_type_erased_host_array_view_t* p
        )

    cdef size_t \
        hipgraph_type_erased_host_array_size(
            const hipgraph_type_erased_host_array_t* p
        )

    cdef data_type_id_t \
        hipgraph_type_erased_host_array_type(
            const hipgraph_type_erased_host_array_t* p
        )

    cdef void* \
        hipgraph_type_erased_host_array_pointer(
            const hipgraph_type_erased_host_array_view_t* p
        )

    # cdef void* \
    #    hipgraph_type_erased_host_array_view_copy(
    #        const hipgraph_resource_handle_t* handle,
    #        hipgraph_type_erased_host_array_view_t* dst,
    #        const hipgraph_type_erased_host_array_view_t* src,
    #        hipgraph_error_t** error
    #    )

    cdef hipgraph_error_code_t \
        hipgraph_type_erased_device_array_view_copy_from_host(
            const hipgraph_resource_handle_t* handle,
            hipgraph_type_erased_device_array_view_t* dst,
            const byte_t* h_src,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_type_erased_device_array_view_copy_to_host(
            const hipgraph_resource_handle_t* handle,
            byte_t* h_dst,
            const hipgraph_type_erased_device_array_view_t* src,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_type_erased_device_array_view_copy(
            const hipgraph_resource_handle_t* handle,
            hipgraph_type_erased_device_array_view_t* dst,
            const hipgraph_type_erased_device_array_view_t* src,
            hipgraph_error_t** error
        )
