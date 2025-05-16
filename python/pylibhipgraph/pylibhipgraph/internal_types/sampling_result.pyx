# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibhipgraph._hipgraph_c.algorithms cimport (
    hipgraph_sample_result_get_destinations,  # deprecated
)
from pylibhipgraph._hipgraph_c.algorithms cimport (
    hipgraph_sample_result_get_hop,  # deprecated
)
from pylibhipgraph._hipgraph_c.algorithms cimport (
    hipgraph_sample_result_get_offsets,  # deprecated
)
from pylibhipgraph._hipgraph_c.algorithms cimport (
    hipgraph_sample_result_get_sources,  # deprecated
)
from pylibhipgraph._hipgraph_c.algorithms cimport (
    hipgraph_sample_result_free,
    hipgraph_sample_result_get_edge_id,
    hipgraph_sample_result_get_edge_type,
    hipgraph_sample_result_get_edge_weight,
    hipgraph_sample_result_get_label_hop_offsets,
    hipgraph_sample_result_get_major_offsets,
    hipgraph_sample_result_get_majors,
    hipgraph_sample_result_get_minors,
    hipgraph_sample_result_get_renumber_map,
    hipgraph_sample_result_get_renumber_map_offsets,
    hipgraph_sample_result_get_start_labels,
    hipgraph_sample_result_t,
)
from pylibhipgraph._hipgraph_c.array cimport hipgraph_type_erased_device_array_view_t
from pylibhipgraph.utils cimport create_cupy_array_view_for_device_ptr


cdef class SamplingResult:
    """
    Cython interface to a hipgraph_sample_result_t pointer. Instances of this
    call will take ownership of the pointer and free it under standard python
    GC rules (ie. when all references to it are no longer present).

    This class provides methods to return non-owning cupy ndarrays for the
    corresponding array members. Returning these cupy arrays increments the ref
    count on the SamplingResult instances from which the cupy arrays are
    referencing.
    """
    def __cinit__(self):
        # This SamplingResult instance owns sample_result_ptr now. It will be
        # freed when this instance is deleted (see __dealloc__())
        self.c_sample_result_ptr = NULL

    def __dealloc__(self):
        if self.c_sample_result_ptr is not NULL:
            hipgraph_sample_result_free(self.c_sample_result_ptr)

    cdef set_ptr(self, hipgraph_sample_result_t* sample_result_ptr):
        self.c_sample_result_ptr = sample_result_ptr

    def get_major_offsets(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")

        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_major_offsets(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_majors(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_majors(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_minors(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_minors(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_sources(self):
        # Deprecated
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_sources(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_destinations(self):
        # Deprecated
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_destinations(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_edge_weights(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_edge_weight(self.c_sample_result_ptr)
        )

        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_indices(self):
        # Deprecated
        return self.get_edge_weights()

    def get_edge_ids(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_edge_id(self.c_sample_result_ptr)
        )

        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_edge_types(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_edge_type(self.c_sample_result_ptr)
        )

        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_batch_ids(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_start_labels(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_label_hop_offsets(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_label_hop_offsets(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    # Deprecated
    def get_offsets(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_offsets(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    # Deprecated
    def get_hop_ids(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_hop(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_renumber_map(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_renumber_map(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_renumber_map_offsets(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef hipgraph_type_erased_device_array_view_t* device_array_view_ptr = (
            hipgraph_sample_result_get_renumber_map_offsets(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)
