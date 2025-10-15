// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "data_type_id.inl.h"
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_device_array_create(const hipgraph_resource_handle_t* handle, size_t n_elems, hipgraph_data_type_id_t dtype, hipgraph_type_erased_device_array_t** array, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_create((const cugraph_resource_handle_t*)handle, n_elems, _hipgraph_to_cugraph_data_type_id_t(dtype), (cugraph_type_erased_device_array_t**)array, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_device_array_create_from_view(const hipgraph_resource_handle_t* handle, const hipgraph_type_erased_device_array_view_t* view, hipgraph_type_erased_device_array_t** array, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_create_from_view((const cugraph_resource_handle_t*)handle, (const cugraph_type_erased_device_array_view_t*)view, (cugraph_type_erased_device_array_t**)array, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT void hipgraph_type_erased_device_array_free(hipgraph_type_erased_device_array_t* p)
{
    cugraph_type_erased_device_array_free((cugraph_type_erased_device_array_t*)p);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_type_erased_device_array_view(hipgraph_type_erased_device_array_t* array)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_type_erased_device_array_view((cugraph_type_erased_device_array_t*)array);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_device_array_view_as_type(hipgraph_type_erased_device_array_t* array, hipgraph_data_type_id_t dtype, hipgraph_type_erased_device_array_view_t** result_view, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_view_as_type((cugraph_type_erased_device_array_t*)array, _hipgraph_to_cugraph_data_type_id_t(dtype), (cugraph_type_erased_device_array_view_t**)result_view, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_type_erased_device_array_view_create(void* pointer, size_t n_elems, hipgraph_data_type_id_t dtype)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_type_erased_device_array_view_create(pointer, n_elems, _hipgraph_to_cugraph_data_type_id_t(dtype));
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_type_erased_device_array_view_free(hipgraph_type_erased_device_array_view_t* p)
{
    cugraph_type_erased_device_array_view_free((cugraph_type_erased_device_array_view_t*)p);
}


HIPGRAPH_EXPORT size_t hipgraph_type_erased_device_array_view_size(const hipgraph_type_erased_device_array_view_t* p)
{
    size_t out;
    out = cugraph_type_erased_device_array_view_size((const cugraph_type_erased_device_array_view_t*)p);
    return (size_t)out;
}


HIPGRAPH_EXPORT hipgraph_data_type_id_t hipgraph_type_erased_device_array_view_type(const hipgraph_type_erased_device_array_view_t* p)
{
    cugraph_data_type_id_t out;
    out = cugraph_type_erased_device_array_view_type((const cugraph_type_erased_device_array_view_t*)p);
    return _cugraph_to_hipgraph_data_type_id_t(out);
}


HIPGRAPH_EXPORT const void* hipgraph_type_erased_device_array_view_pointer(const hipgraph_type_erased_device_array_view_t* p)
{
    const void* out;
    out = cugraph_type_erased_device_array_view_pointer((const cugraph_type_erased_device_array_view_t*)p);
    return (const void*)out;
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_host_array_create(const hipgraph_resource_handle_t* handle, size_t n_elems, hipgraph_data_type_id_t dtype, hipgraph_type_erased_host_array_t** array, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_host_array_create((const cugraph_resource_handle_t*)handle, n_elems, _hipgraph_to_cugraph_data_type_id_t(dtype), (cugraph_type_erased_host_array_t**)array, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT void hipgraph_type_erased_host_array_free(hipgraph_type_erased_host_array_t* p)
{
    cugraph_type_erased_host_array_free((cugraph_type_erased_host_array_t*)p);
}


HIPGRAPH_EXPORT hipgraph_type_erased_host_array_view_t* hipgraph_type_erased_host_array_view(hipgraph_type_erased_host_array_t* array)
{
    cugraph_type_erased_host_array_view_t* out;
    out = cugraph_type_erased_host_array_view((cugraph_type_erased_host_array_t*)array);
    return (hipgraph_type_erased_host_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_host_array_view_t* hipgraph_type_erased_host_array_view_create(void* pointer, size_t n_elems, hipgraph_data_type_id_t dtype)
{
    cugraph_type_erased_host_array_view_t* out;
    out = cugraph_type_erased_host_array_view_create(pointer, n_elems, _hipgraph_to_cugraph_data_type_id_t(dtype));
    return (hipgraph_type_erased_host_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_type_erased_host_array_view_free(hipgraph_type_erased_host_array_view_t* p)
{
    cugraph_type_erased_host_array_view_free((cugraph_type_erased_host_array_view_t*)p);
}


HIPGRAPH_EXPORT size_t hipgraph_type_erased_host_array_size(const hipgraph_type_erased_host_array_view_t* p)
{
    size_t out;
    out = cugraph_type_erased_host_array_size((const cugraph_type_erased_host_array_view_t*)p);
    return (size_t)out;
}


HIPGRAPH_EXPORT hipgraph_data_type_id_t hipgraph_type_erased_host_array_type(const hipgraph_type_erased_host_array_view_t* p)
{
    cugraph_data_type_id_t out;
    out = cugraph_type_erased_host_array_type((const cugraph_type_erased_host_array_view_t*)p);
    return _cugraph_to_hipgraph_data_type_id_t(out);
}


HIPGRAPH_EXPORT void* hipgraph_type_erased_host_array_pointer(const hipgraph_type_erased_host_array_view_t* p)
{
    void* out;
    out = cugraph_type_erased_host_array_pointer((const cugraph_type_erased_host_array_view_t*)p);
    return (void*)out;
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_host_array_view_copy(const hipgraph_resource_handle_t* handle, hipgraph_type_erased_host_array_view_t* dst, const hipgraph_type_erased_host_array_view_t* src, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_host_array_view_copy((const cugraph_resource_handle_t*)handle, (cugraph_type_erased_host_array_view_t*)dst, (const cugraph_type_erased_host_array_view_t*)src, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_device_array_view_copy_from_host(const hipgraph_resource_handle_t* handle, hipgraph_type_erased_device_array_view_t* dst, const char* h_src, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_view_copy_from_host((const cugraph_resource_handle_t*)handle, (cugraph_type_erased_device_array_view_t*)dst, h_src, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_device_array_view_copy_to_host(const hipgraph_resource_handle_t* handle, char* h_dst, const hipgraph_type_erased_device_array_view_t* src, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_view_copy_to_host((const cugraph_resource_handle_t*)handle, h_dst, (const cugraph_type_erased_device_array_view_t*)src, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_device_array_view_copy(const hipgraph_resource_handle_t* handle, hipgraph_type_erased_device_array_view_t* dst, const hipgraph_type_erased_device_array_view_t* src, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_view_copy((const cugraph_resource_handle_t*)handle, (cugraph_type_erased_device_array_view_t*)dst, (const cugraph_type_erased_device_array_view_t*)src, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}




