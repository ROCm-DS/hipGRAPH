// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "common.h"
#include <cugraph_c/array.h>
#include "hipgraph/hipgraph_c/array.h"

hipgraph_error_code_t
    hipgraph_type_erased_device_array_create(const hipgraph_resource_handle_t*     handle,
                                             size_t                                n_elems,
                                             hipgraph_data_type_id_t               dtype,
                                             hipgraph_type_erased_device_array_t** array,
                                             hipgraph_error_t**                    error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_create((const cugraph_resource_handle_t*)handle,
                                                  n_elems,
                                                  (cugraph_data_type_id_t)dtype,
                                                  (cugraph_type_erased_device_array_t**)array,
                                                  (cugraph_error_t**)error);
    return (hipgraph_error_code_t)out;
};

hipgraph_error_code_t hipgraph_type_erased_device_array_create_from_view(
    const hipgraph_resource_handle_t*               handle,
    const hipgraph_type_erased_device_array_view_t* view,
    hipgraph_type_erased_device_array_t**           array,
    hipgraph_error_t**                              error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_create_from_view(
        (const cugraph_resource_handle_t*)handle,
        (const cugraph_type_erased_device_array_view_t*)view,
        (cugraph_type_erased_device_array_t**)array,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)out;
};

void hipgraph_type_erased_device_array_free(hipgraph_type_erased_device_array_t* p)
{
    cugraph_type_erased_device_array_free((cugraph_type_erased_device_array_t*)p);
};

#ifdef HIPGRAPH_EXCLUDE
void* hipgraph_type_erased_device_array_release(hipgraph_type_erased_device_array_t* p);
#endif

hipgraph_type_erased_device_array_view_t*
    hipgraph_type_erased_device_array_view(hipgraph_type_erased_device_array_t* array)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_type_erased_device_array_view((cugraph_type_erased_device_array_t*)array);
    return (hipgraph_type_erased_device_array_view_t*)out;
};

hipgraph_error_code_t hipgraph_type_erased_device_array_view_as_type(
    hipgraph_type_erased_device_array_t*       array,
    hipgraph_data_type_id_t                    dtype,
    hipgraph_type_erased_device_array_view_t** result_view,
    hipgraph_error_t**                         error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_view_as_type(
        (cugraph_type_erased_device_array_t*)array,
        (cugraph_data_type_id_t)dtype,
        (cugraph_type_erased_device_array_view_t**)result_view,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)out;
};

hipgraph_type_erased_device_array_view_t* hipgraph_type_erased_device_array_view_create(
    void* pointer, size_t n_elems, hipgraph_data_type_id_t dtype)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_type_erased_device_array_view_create(
        pointer, n_elems, (cugraph_data_type_id_t)dtype);
    return (hipgraph_type_erased_device_array_view_t*)out;
};

void hipgraph_type_erased_device_array_view_free(hipgraph_type_erased_device_array_view_t* p)
{
    cugraph_type_erased_device_array_view_free((cugraph_type_erased_device_array_view_t*)p);
};

size_t
    hipgraph_type_erased_device_array_view_size(const hipgraph_type_erased_device_array_view_t* p)
{
    size_t out;
    out = cugraph_type_erased_device_array_view_size(
        (const cugraph_type_erased_device_array_view_t*)p);
    return out;
};

hipgraph_data_type_id_t
    hipgraph_type_erased_device_array_view_type(const hipgraph_type_erased_device_array_view_t* p)
{
    cugraph_data_type_id_t out;
    out = cugraph_type_erased_device_array_view_type(
        (const cugraph_type_erased_device_array_view_t*)p);
    return (hipgraph_data_type_id_t)out;
};

const void* hipgraph_type_erased_device_array_view_pointer(
    const hipgraph_type_erased_device_array_view_t* p)
{
    const void* out = cugraph_type_erased_device_array_view_pointer(
        (const cugraph_type_erased_device_array_view_t*)p);
    return out;
};

hipgraph_error_code_t
    hipgraph_type_erased_host_array_create(const hipgraph_resource_handle_t*   handle,
                                           size_t                              n_elems,
                                           hipgraph_data_type_id_t             dtype,
                                           hipgraph_type_erased_host_array_t** array,
                                           hipgraph_error_t**                  error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_host_array_create((const cugraph_resource_handle_t*)handle,
                                                n_elems,
                                                (cugraph_data_type_id_t)dtype,
                                                (cugraph_type_erased_host_array_t**)array,
                                                (cugraph_error_t**)error);
    return (hipgraph_error_code_t)out;
};

void hipgraph_type_erased_host_array_free(hipgraph_type_erased_host_array_t* p)
{
    cugraph_type_erased_host_array_free((cugraph_type_erased_host_array_t*)p);
};

#ifdef HIPGRAPH_EXCLUDE
void* hipgraph_type_erased_host_array_release(hipgraph_type_erased_host_array_t* p);
#endif

hipgraph_type_erased_host_array_view_t*
    hipgraph_type_erased_host_array_view(hipgraph_type_erased_host_array_t* array)
{
    cugraph_type_erased_host_array_view_t* out;
    out = cugraph_type_erased_host_array_view((cugraph_type_erased_host_array_t*)array);
    return (hipgraph_type_erased_host_array_view_t*)out;
};

hipgraph_type_erased_host_array_view_t* hipgraph_type_erased_host_array_view_create(
    void* pointer, size_t n_elems, hipgraph_data_type_id_t dtype)
{
    cugraph_type_erased_host_array_view_t* out;
    out = cugraph_type_erased_host_array_view_create(
        pointer, n_elems, (cugraph_data_type_id_t)dtype);
    return (hipgraph_type_erased_host_array_view_t*)out;
};

void hipgraph_type_erased_host_array_view_free(hipgraph_type_erased_host_array_view_t* p)
{
    cugraph_type_erased_host_array_view_free((cugraph_type_erased_host_array_view_t*)p);
};

size_t hipgraph_type_erased_host_array_size(const hipgraph_type_erased_host_array_view_t* p)
{
    size_t out;
    out = cugraph_type_erased_host_array_size((const cugraph_type_erased_host_array_view_t*)p);
    return out;
};

#if 0
/* Not implemented in cugraph_c. */
hipgraph_data_type_id_t
    hipgraph_type_erased_host_array_type(const hipgraph_type_erased_host_array_view_t* p)
{
    cugraph_data_type_id_t out;
    out = cugraph_type_erased_host_array_type((const cugraph_type_erased_host_array_view_t*)p);
    return (hipgraph_data_type_id_t)out;
};
#endif

void* hipgraph_type_erased_host_array_pointer(const hipgraph_type_erased_host_array_view_t* p)
{
    void* out;
    out = cugraph_type_erased_host_array_pointer((const cugraph_type_erased_host_array_view_t*)p);
    return out;
};

hipgraph_error_code_t
    hipgraph_type_erased_host_array_view_copy(const hipgraph_resource_handle_t*             handle,
                                              hipgraph_type_erased_host_array_view_t*       dst,
                                              const hipgraph_type_erased_host_array_view_t* src,
                                              hipgraph_error_t**                            error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_host_array_view_copy(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_type_erased_host_array_view_t*)dst,
        (const cugraph_type_erased_host_array_view_t*)src,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)out;
};

hipgraph_error_code_t hipgraph_type_erased_device_array_view_copy_from_host(
    const hipgraph_resource_handle_t*         handle,
    hipgraph_type_erased_device_array_view_t* dst,
    const hipgraph_byte_t*                    h_src,
    hipgraph_error_t**                        error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_view_copy_from_host(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_type_erased_device_array_view_t*)dst,
        (const byte_t*)h_src,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)out;
};

hipgraph_error_code_t hipgraph_type_erased_device_array_view_copy_to_host(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_byte_t*                                h_dst,
    const hipgraph_type_erased_device_array_view_t* src,
    hipgraph_error_t**                              error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_view_copy_to_host(
        (const cugraph_resource_handle_t*)handle,
        (byte_t*)h_dst,
        (const cugraph_type_erased_device_array_view_t*)src,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)out;
};

hipgraph_error_code_t
    hipgraph_type_erased_device_array_view_copy(const hipgraph_resource_handle_t*         handle,
                                                hipgraph_type_erased_device_array_view_t* dst,
                                                const hipgraph_type_erased_device_array_view_t* src,
                                                hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_type_erased_device_array_view_copy(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_type_erased_device_array_view_t*)dst,
        (const cugraph_type_erased_device_array_view_t*)src,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)out;
};
