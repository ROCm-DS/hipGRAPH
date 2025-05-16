# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t

import cupy
import numpy

from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_copy,
    hipgraph_type_erased_device_array_view_create,
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_pointer,
    hipgraph_type_erased_device_array_view_size,
    hipgraph_type_erased_device_array_view_type,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_free, hipgraph_error_message


# FIXME: add tests for this
cdef assert_success(hipgraph_error_code_t code,
                    hipgraph_error_t* err,
                    api_name):
    if code != hipgraph_error_code_t.HIPGRAPH_SUCCESS:
        c_error = hipgraph_error_message(err)
        if isinstance(c_error, bytes):
            c_error = c_error.decode()
        else:
            c_error = str(c_error)

        hipgraph_error_free(err)

        if code == hipgraph_error_code_t.HIPGRAPH_UNKNOWN_ERROR:
            code_str = "HIPGRAPH_UNKNOWN_ERROR"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise RuntimeError(error_msg)
        elif code == hipgraph_error_code_t.HIPGRAPH_INVALID_HANDLE:
            code_str = "HIPGRAPH_INVALID_HANDLE"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise ValueError(error_msg)
        elif code == hipgraph_error_code_t.HIPGRAPH_ALLOC_ERROR:
            code_str = "HIPGRAPH_ALLOC_ERROR"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise MemoryError(error_msg)
        elif code == hipgraph_error_code_t.HIPGRAPH_INVALID_INPUT:
            code_str = "HIPGRAPH_INVALID_INPUT"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise ValueError(error_msg)
        elif code == hipgraph_error_code_t.HIPGRAPH_NOT_IMPLEMENTED:
            code_str = "HIPGRAPH_NOT_IMPLEMENTED"
            error_msg = f"non-success value returned from {api_name}: {code_str}\ "\
                        f"{c_error}"
            raise NotImplementedError(error_msg)
        elif code == hipgraph_error_code_t.HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION:
            code_str = "HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise ValueError(error_msg)
        else:
            code_str = "unknown error code"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise RuntimeError(error_msg)


cdef assert_CAI_type(obj, var_name, allow_None=False):
    if allow_None:
        if obj is None:
            return
        msg = f"{var_name} must be None or support __cuda_array_interface__"
    else:
        msg = f"{var_name} does not support __cuda_array_interface__"

    if not(hasattr(obj, "__cuda_array_interface__")):
        raise TypeError(msg)


cdef assert_AI_type(obj, var_name, allow_None=False):
    if allow_None:
        if obj is None:
            return
        msg = f"{var_name} must be None or support __array_interface__"
    else:
        msg = f"{var_name} does not support __array_interface__"

    if not(hasattr(obj, "__array_interface__")):
        raise TypeError(msg)


cdef get_numpy_type_from_c_type(data_type_id_t c_type):
    if c_type == data_type_id_t.INT32:
        return numpy.int32
    elif c_type == data_type_id_t.INT64:
        return numpy.int64
    elif c_type == data_type_id_t.FLOAT32:
        return numpy.float32
    elif c_type == data_type_id_t.FLOAT64:
        return numpy.float64
    elif c_type == data_type_id_t.SIZE_T:
        return numpy.int64
    else:
        raise RuntimeError("Internal error: got invalid data type enum value "
                           f"from C: {c_type}")


cdef get_c_type_from_numpy_type(numpy_type):
    dt = numpy.dtype(numpy_type)
    if dt == numpy.int32:
        return data_type_id_t.INT32
    elif dt == numpy.int64:
        return data_type_id_t.INT64
    elif dt == numpy.float32:
        return data_type_id_t.FLOAT32
    elif dt == numpy.float64:
        return data_type_id_t.FLOAT64
    else:
        raise RuntimeError("Internal error: got invalid data type enum value "
                          f"from Numpy: {numpy_type}")

cdef get_c_weight_type_from_numpy_edge_ids_type(numpy_type):
    if numpy_type == numpy.int32:
        return data_type_id_t.FLOAT32
    else:
        return data_type_id_t.FLOAT64

cdef get_numpy_edge_ids_type_from_c_weight_type(data_type_id_t c_weight_type):
    if c_weight_type == data_type_id_t.FLOAT32:
        return numpy.int32
    else:
        return numpy.int64


cdef copy_to_cupy_array(
   hipgraph_resource_handle_t* c_resource_handle_ptr,
   hipgraph_type_erased_device_array_view_t* device_array_view_ptr):
    """
    Copy the contents from a device array view as returned by various hipgraph_*
    APIs to a new cupy device array, typically intended to be used as a return
    value from pylibhipgraph APIs.
    """
    cdef c_type = hipgraph_type_erased_device_array_view_type(
        device_array_view_ptr)
    array_size = hipgraph_type_erased_device_array_view_size(
        device_array_view_ptr)

    cupy_array = cupy.zeros(
        array_size, dtype=get_numpy_type_from_c_type(c_type))

    cdef uintptr_t cupy_array_ptr = \
        cupy_array.__cuda_array_interface__["data"][0]

    cdef hipgraph_type_erased_device_array_view_t* cupy_array_view_ptr = \
        hipgraph_type_erased_device_array_view_create(
            <void*>cupy_array_ptr, array_size, c_type)

    cdef hipgraph_error_t* error_ptr
    error_code = hipgraph_type_erased_device_array_view_copy(
        c_resource_handle_ptr,
        cupy_array_view_ptr,
        device_array_view_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr,
                   "hipgraph_type_erased_device_array_view_copy")

    hipgraph_type_erased_device_array_view_free(device_array_view_ptr)

    return cupy_array

cdef copy_to_cupy_array_ids(
   hipgraph_resource_handle_t* c_resource_handle_ptr,
   hipgraph_type_erased_device_array_view_t* device_array_view_ptr):
    """
    Copy the contents from a device array view as returned by various hipgraph_*
    APIs to a new cupy device array, typically intended to be used as a return
    value from pylibhipgraph APIs then convert float to int
    """
    cdef c_type = hipgraph_type_erased_device_array_view_type(
        device_array_view_ptr)

    array_size = hipgraph_type_erased_device_array_view_size(
        device_array_view_ptr)

    cupy_array = cupy.zeros(
        array_size, dtype=get_numpy_edge_ids_type_from_c_weight_type(c_type))

    cdef uintptr_t cupy_array_ptr = \
        cupy_array.__cuda_array_interface__["data"][0]

    cdef hipgraph_type_erased_device_array_view_t* cupy_array_view_ptr = \
        hipgraph_type_erased_device_array_view_create(
            <void*>cupy_array_ptr, array_size, get_c_type_from_numpy_type(cupy_array.dtype))

    cdef hipgraph_error_t* error_ptr
    error_code = hipgraph_type_erased_device_array_view_copy(
        c_resource_handle_ptr,
        cupy_array_view_ptr,
        device_array_view_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr,
                   "hipgraph_type_erased_device_array_view_copy")

    hipgraph_type_erased_device_array_view_free(device_array_view_ptr)

    return cupy_array

cdef hipgraph_type_erased_device_array_view_t* \
    create_hipgraph_type_erased_device_array_view_from_py_obj(python_obj):
        cdef uintptr_t cai_ptr = <uintptr_t>NULL
        cdef hipgraph_type_erased_device_array_view_t* view_ptr = NULL
        if python_obj is not None:
            cai_ptr = python_obj.__cuda_array_interface__["data"][0]
            view_ptr = hipgraph_type_erased_device_array_view_create(
                <void*>cai_ptr,
                len(python_obj),
                get_c_type_from_numpy_type(python_obj.dtype))

        return view_ptr

cdef create_cupy_array_view_for_device_ptr(
    hipgraph_type_erased_device_array_view_t* device_array_view_ptr,
    owning_py_object):

    if device_array_view_ptr == NULL:
        raise ValueError("device_array_view_ptr cannot be NULL")

    cdef c_type = hipgraph_type_erased_device_array_view_type(
        device_array_view_ptr)
    array_size = hipgraph_type_erased_device_array_view_size(
        device_array_view_ptr)
    dtype = get_numpy_type_from_c_type(c_type)

    cdef uintptr_t ptr_value = \
        <uintptr_t> hipgraph_type_erased_device_array_view_pointer(device_array_view_ptr)

    if ptr_value == <uintptr_t> NULL:
        # For the case of a NULL ptr, just create a new empty ndarray of the
        # appropriate type. This will not be associated with the
        # owning_py_object, but will still be garbage collected correctly.
        cupy_array = cupy.ndarray(0, dtype=dtype)

    else:
        # cupy.cuda.UnownedMemory takes a reference to an owning python object
        # which is used to increment the refcount on the owning python object.
        # This prevents the owning python object from being garbage collected
        # and having the memory freed when there are instances of the
        # cupy_array still in use that need the memory.  When the cupy_array
        # instance returned here is deleted, it will decrement the refcount on
        # the owning python object, and when that refcount reaches zero the
        # owning python object will be garbage collected and the memory freed.
        cpmem = cupy.cuda.UnownedMemory(ptr_value,
                                        array_size,
                                        owning_py_object)
        cpmem_ptr = cupy.cuda.MemoryPointer(cpmem, 0)
        cupy_array = cupy.ndarray(
            array_size,
            dtype=dtype,
            memptr=cpmem_ptr)

    return cupy_array
