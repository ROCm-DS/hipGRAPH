// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */

/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.h"
#include <rocgraph/rocgraph.h>
#include "hipgraph/hipgraph_c/array.h"

hipgraph_error_code_t
    hipgraph_type_erased_device_array_create(const hipgraph_resource_handle_t*     handle,
                                             size_t                                n_elems,
                                             hipgraph_data_type_id_t               dtype,
                                             hipgraph_type_erased_device_array_t** array,
                                             hipgraph_error_t**                    error)
{
    rocgraph_data_type_id rg_dtype = hipgraph_data_type_id_t2rocgraph_data_type_id(dtype);
    if(hghelper_rocgraph_data_type_id_is_invalid(rg_dtype))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_type_erased_device_array_create((const rocgraph_handle_t*)handle,
                                                   n_elems,
                                                   rg_dtype,
                                                   (rocgraph_type_erased_device_array_t**)array,
                                                   (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_type_erased_device_array_create_from_view(
    const hipgraph_resource_handle_t*               handle,
    const hipgraph_type_erased_device_array_view_t* view,
    hipgraph_type_erased_device_array_t**           array,
    hipgraph_error_t**                              error)
{
    rocgraph_status rg_status = rocgraph_type_erased_device_array_create_from_view(
        (const rocgraph_handle_t*)handle,
        (const rocgraph_type_erased_device_array_view_t*)view,
        (rocgraph_type_erased_device_array_t**)array,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

void hipgraph_type_erased_device_array_free(hipgraph_type_erased_device_array_t* p)
{
    rocgraph_type_erased_device_array_free((rocgraph_type_erased_device_array_t*)p);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_type_erased_device_array_view(hipgraph_type_erased_device_array_t* array)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_type_erased_device_array_view(
        (rocgraph_type_erased_device_array_t*)array);
}

hipgraph_error_code_t hipgraph_type_erased_device_array_view_as_type(
    hipgraph_type_erased_device_array_t*       array,
    hipgraph_data_type_id_t                    dtype,
    hipgraph_type_erased_device_array_view_t** result_view,
    hipgraph_error_t**                         error)
{
    rocgraph_data_type_id rg_dtype = hipgraph_data_type_id_t2rocgraph_data_type_id(dtype);
    if(hghelper_rocgraph_data_type_id_is_invalid(rg_dtype))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_type_erased_device_array_view_as_type(
        (rocgraph_type_erased_device_array_t*)array,
        rg_dtype,
        (rocgraph_type_erased_device_array_view_t**)result_view,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t* hipgraph_type_erased_device_array_view_create(
    void* pointer, size_t n_elems, hipgraph_data_type_id_t dtype)
{
    rocgraph_data_type_id rg_dtype = hipgraph_data_type_id_t2rocgraph_data_type_id(dtype);
    if(hghelper_rocgraph_data_type_id_is_invalid(rg_dtype))
        return NULL;
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_type_erased_device_array_view_create(
        pointer, n_elems, rg_dtype);
}

void hipgraph_type_erased_device_array_view_free(hipgraph_type_erased_device_array_view_t* p)
{
    rocgraph_type_erased_device_array_view_free((rocgraph_type_erased_device_array_view_t*)p);
}

size_t
    hipgraph_type_erased_device_array_view_size(const hipgraph_type_erased_device_array_view_t* p)
{
    return (size_t)rocgraph_type_erased_device_array_view_size(
        (const rocgraph_type_erased_device_array_view_t*)p);
}

hipgraph_data_type_id_t
    hipgraph_type_erased_device_array_view_type(const hipgraph_type_erased_device_array_view_t* p)
{
    return rocgraph_data_type_id2hipgraph_data_type_id_t(
        rocgraph_type_erased_device_array_view_type(
            (const rocgraph_type_erased_device_array_view_t*)p));
}

const void* hipgraph_type_erased_device_array_view_pointer(
    const hipgraph_type_erased_device_array_view_t* p)
{
    return (const void*)rocgraph_type_erased_device_array_view_pointer(
        (const rocgraph_type_erased_device_array_view_t*)p);
}

hipgraph_error_code_t
    hipgraph_type_erased_host_array_create(const hipgraph_resource_handle_t*   handle,
                                           size_t                              n_elems,
                                           hipgraph_data_type_id_t             dtype,
                                           hipgraph_type_erased_host_array_t** array,
                                           hipgraph_error_t**                  error)
{
    rocgraph_data_type_id rg_dtype = hipgraph_data_type_id_t2rocgraph_data_type_id(dtype);
    if(hghelper_rocgraph_data_type_id_is_invalid(rg_dtype))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_type_erased_host_array_create((const rocgraph_handle_t*)handle,
                                                 n_elems,
                                                 rg_dtype,
                                                 (rocgraph_type_erased_host_array_t**)array,
                                                 (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

void hipgraph_type_erased_host_array_free(hipgraph_type_erased_host_array_t* p)
{
    rocgraph_type_erased_host_array_free((rocgraph_type_erased_host_array_t*)p);
}

hipgraph_type_erased_host_array_view_t*
    hipgraph_type_erased_host_array_view(hipgraph_type_erased_host_array_t* array)
{
    return (hipgraph_type_erased_host_array_view_t*)rocgraph_type_erased_host_array_view(
        (rocgraph_type_erased_host_array_t*)array);
}

hipgraph_type_erased_host_array_view_t* hipgraph_type_erased_host_array_view_create(
    void* pointer, size_t n_elems, hipgraph_data_type_id_t dtype)
{
    rocgraph_data_type_id rg_dtype = hipgraph_data_type_id_t2rocgraph_data_type_id(dtype);
    if(hghelper_rocgraph_data_type_id_is_invalid(rg_dtype))
        return NULL;
    return (hipgraph_type_erased_host_array_view_t*)rocgraph_type_erased_host_array_view_create(
        pointer, n_elems, rg_dtype);
}

void hipgraph_type_erased_host_array_view_free(hipgraph_type_erased_host_array_view_t* p)
{
    rocgraph_type_erased_host_array_view_free((rocgraph_type_erased_host_array_view_t*)p);
}

size_t hipgraph_type_erased_host_array_size(const hipgraph_type_erased_host_array_view_t* p)
{
    return (size_t)rocgraph_type_erased_host_array_size(
        (const rocgraph_type_erased_host_array_view_t*)p);
}
#if 0
hipgraph_data_type_id_t
hipgraph_type_erased_host_array_type(const hipgraph_type_erased_host_array_view_t* p)
{
  return (hipgraph_data_type_id_t)rocgraph_type_erased_host_array_type(
								       (const rocgraph_type_erased_host_array_view_t*)p));
}
#endif
void* hipgraph_type_erased_host_array_pointer(const hipgraph_type_erased_host_array_view_t* p)
{
    return (void*)rocgraph_type_erased_host_array_pointer(
        (const rocgraph_type_erased_host_array_view_t*)p);
}

hipgraph_error_code_t
    hipgraph_type_erased_host_array_view_copy(const hipgraph_resource_handle_t*             handle,
                                              hipgraph_type_erased_host_array_view_t*       dst,
                                              const hipgraph_type_erased_host_array_view_t* src,
                                              hipgraph_error_t**                            error)
{
    rocgraph_status rg_status = rocgraph_type_erased_host_array_view_copy(
        (const rocgraph_handle_t*)handle,
        (rocgraph_type_erased_host_array_view_t*)dst,
        (const rocgraph_type_erased_host_array_view_t*)src,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_type_erased_device_array_view_copy_from_host(
    const hipgraph_resource_handle_t*         handle,
    hipgraph_type_erased_device_array_view_t* dst,
    const hipgraph_byte_t*                    h_src,
    hipgraph_error_t**                        error)
{
    rocgraph_status rg_status = rocgraph_type_erased_device_array_view_copy_from_host(
        (const rocgraph_handle_t*)handle,
        (rocgraph_type_erased_device_array_view_t*)dst,
        h_src,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_type_erased_device_array_view_copy_to_host(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_byte_t*                                h_dst,
    const hipgraph_type_erased_device_array_view_t* src,
    hipgraph_error_t**                              error)
{
    rocgraph_status rg_status = rocgraph_type_erased_device_array_view_copy_to_host(
        (const rocgraph_handle_t*)handle,
        h_dst,
        (const rocgraph_type_erased_device_array_view_t*)src,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_type_erased_device_array_view_copy(const hipgraph_resource_handle_t*         handle,
                                                hipgraph_type_erased_device_array_view_t* dst,
                                                const hipgraph_type_erased_device_array_view_t* src,
                                                hipgraph_error_t** error)
{
    rocgraph_status rg_status = rocgraph_type_erased_device_array_view_copy(
        (const rocgraph_handle_t*)handle,
        (rocgraph_type_erased_device_array_view_t*)dst,
        (const rocgraph_type_erased_device_array_view_t*)src,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}
