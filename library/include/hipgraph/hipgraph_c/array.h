// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
 *
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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
 * ************************************************************************ */
#pragma once

#include "hipgraph/hipgraph_c/resource_handle.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int32_t align_;
} hipgraph_type_erased_device_array_t;

typedef struct
{
    int32_t align_;
} hipgraph_type_erased_device_array_view_t;

typedef struct
{
    int32_t align_;
} hipgraph_type_erased_host_array_t;

typedef struct
{
    int32_t align_;
} hipgraph_type_erased_host_array_view_t;

/**
 * @brief     Create a type erased device array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @param [out] array       Pointer to the location to store the pointer to the device array
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_type_erased_device_array_create(const hipgraph_resource_handle_t*     handle,
                                             size_t                                n_elems,
                                             hipgraph_data_type_id_t               dtype,
                                             hipgraph_type_erased_device_array_t** array,
                                             hipgraph_error_t**                    error);

/**
 * @brief     Create a type erased device array from a view
 *
 * Copies the data from the view into the new device array
 *
 * @param [in]  handle Handle for accessing resources
 * @param [in]  view   Type erased device array view to copy from
 * @param [out] array  Pointer to the location to store the pointer to the device array
 * @param [out] error  Pointer to an error object storing details of any error.  Will
 *                     be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_device_array_create_from_view(
    const hipgraph_resource_handle_t*               handle,
    const hipgraph_type_erased_device_array_view_t* view,
    hipgraph_type_erased_device_array_t**           array,
    hipgraph_error_t**                              error);

/**
 * @brief    Destroy a type erased device array
 *
 * @param [in]  p    Pointer to the type erased device array
 */
HIPGRAPH_EXPORT void hipgraph_type_erased_device_array_free(hipgraph_type_erased_device_array_t* p);

#if 0
// FIXME: Not implemented, need to discuss if this can work.  We will either implement
//        this later or delete it from the interface once we resolve how to handle this
/**
 * @brief    Release the raw pointer of the type erased device array
 *
 * The caller is now responsible for freeing the device pointer
 *
 * @param [in]  p    Pointer to the type erased device array
 * @return Pointer (device memory) for the data in the array
 */
HIPGRAPH_EXPORT void* hipgraph_type_erased_device_array_release(hipgraph_type_erased_device_array_t* p);
#endif

/**
 * @brief    Create a type erased device array view from
 *           a type erased device array
 *
 * @param [in]  array       Pointer to the type erased device array
 * @return Pointer to the view of the host array
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_type_erased_device_array_view(hipgraph_type_erased_device_array_t* array);

/**
 * @brief Create a type erased device array view with a different type
 *
 *    Create a type erased device array view from
 *    a type erased device array treating the underlying
 *    pointer as a different type.
 *
 *    Note: This is only viable when the underlying types are the same size.  That
 *    is, you can switch between INT32 and FLOAT32, or between INT64 and FLOAT64.
 *    But if the types are different sizes this will be an error.
 *
 * @param [in]  array        Pointer to the type erased device array
 * @param [in]  dtype        The type to cast the pointer to
 * @param [out] result_view  Address where to put the allocated device view
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_device_array_view_as_type(
    hipgraph_type_erased_device_array_t*       array,
    hipgraph_data_type_id_t                    dtype,
    hipgraph_type_erased_device_array_view_t** result_view,
    hipgraph_error_t**                         error);

/**
 * @brief    Create a type erased device array view from
 *           a raw device pointer.
 *
 * @param [in]  pointer     Raw device pointer
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @return Pointer to the view of the host array
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_type_erased_device_array_view_create(void*                   pointer,
                                                  size_t                  n_elems,
                                                  hipgraph_data_type_id_t dtype);

/**
 * @brief    Destroy a type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 */
HIPGRAPH_EXPORT void
    hipgraph_type_erased_device_array_view_free(hipgraph_type_erased_device_array_view_t* p);

/**
 * @brief    Get the size of a type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 * @return The number of elements in the array
 */
HIPGRAPH_EXPORT size_t
    hipgraph_type_erased_device_array_view_size(const hipgraph_type_erased_device_array_view_t* p);

/**
 * @brief    Get the type of a type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 * @return The type of the elements in the array
 */
HIPGRAPH_EXPORT hipgraph_data_type_id_t
    hipgraph_type_erased_device_array_view_type(const hipgraph_type_erased_device_array_view_t* p);

/**
 * @brief    Get the raw pointer of the type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 * @return Pointer (device memory) for the data in the array
 */
HIPGRAPH_EXPORT const void* hipgraph_type_erased_device_array_view_pointer(
    const hipgraph_type_erased_device_array_view_t* p);

/**
 * @brief     Create a type erased host array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @param [out] array       Pointer to the location to store the pointer to the host array
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_type_erased_host_array_create(const hipgraph_resource_handle_t*   handle,
                                           size_t                              n_elems,
                                           hipgraph_data_type_id_t             dtype,
                                           hipgraph_type_erased_host_array_t** array,
                                           hipgraph_error_t**                  error);

/**
 * @brief    Destroy a type erased host array
 *
 * @param [in]  p    Pointer to the type erased host array
 */
HIPGRAPH_EXPORT void hipgraph_type_erased_host_array_free(hipgraph_type_erased_host_array_t* p);

#if 0
// FIXME: Not implemented, need to discuss if this can work.  We will either implement
//        this later or delete it from the interface once we resolve how to handle this
/**
 * @brief    Release the raw pointer of the type erased host array
 *
 * The caller is now responsible for freeing the host pointer
 *
 * @param [in]  p    Pointer to the type erased host array
 * @return Pointer (host memory) for the data in the array
 */
void* hipgraph_type_erased_host_array_release(hipgraph_type_erased_host_array_t* p);
#endif

/**
 * @brief    Create a type erased host array view from
 *           a type erased host array
 *
 * @param [in]  array       Pointer to the type erased host array
 * @return Pointer to the view of the host array
 */
HIPGRAPH_EXPORT hipgraph_type_erased_host_array_view_t*
    hipgraph_type_erased_host_array_view(hipgraph_type_erased_host_array_t* array);

/**
 * @brief    Create a type erased host array view from
 *           a raw host pointer.
 *
 * @param [in]  pointer     Raw host pointer
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @return pointer to the view of the host array
 */
HIPGRAPH_EXPORT hipgraph_type_erased_host_array_view_t* hipgraph_type_erased_host_array_view_create(
    void* pointer, size_t n_elems, hipgraph_data_type_id_t dtype);

/**
 * @brief    Destroy a type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 */
HIPGRAPH_EXPORT void
    hipgraph_type_erased_host_array_view_free(hipgraph_type_erased_host_array_view_t* p);

/**
 * @brief    Get the size of a type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 * @return The number of elements in the array
 */
HIPGRAPH_EXPORT size_t
    hipgraph_type_erased_host_array_size(const hipgraph_type_erased_host_array_view_t* p);

/**
 * @brief    Get the type of a type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 * @return The type of the elements in the array
 */
HIPGRAPH_EXPORT hipgraph_data_type_id_t
    hipgraph_type_erased_host_array_type(const hipgraph_type_erased_host_array_view_t* p);

/**
 * @brief    Get the raw pointer of the type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 * @return Pointer (host memory) for the data in the array
 */
HIPGRAPH_EXPORT void*
    hipgraph_type_erased_host_array_pointer(const hipgraph_type_erased_host_array_view_t* p);

/**
 * @brief    Copy data between two type erased device array views
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] dst         Pointer to type erased host array view destination
 * @param [in]  src         Pointer to type erased host array view source
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_type_erased_host_array_view_copy(const hipgraph_resource_handle_t*             handle,
                                              hipgraph_type_erased_host_array_view_t*       dst,
                                              const hipgraph_type_erased_host_array_view_t* src,
                                              hipgraph_error_t**                            error);

/**
 * @brief    Copy data from host to a type erased device array view
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] dst         Pointer to the type erased device array view
 * @param [in]  h_src       Pointer to host array to copy into device memory
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_device_array_view_copy_from_host(
    const hipgraph_resource_handle_t*         handle,
    hipgraph_type_erased_device_array_view_t* dst,
    const hipgraph_byte_t*                    h_src,
    hipgraph_error_t**                        error);

/**
 * @brief    Copy data from device to a type erased host array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] h_dst       Pointer to host array
 * @param [in]  src         Pointer to the type erased device array view source
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_type_erased_device_array_view_copy_to_host(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_byte_t*                                h_dst,
    const hipgraph_type_erased_device_array_view_t* src,
    hipgraph_error_t**                              error);

/**
 * @brief    Copy data between two type erased device array views
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] dst         Pointer to type erased device array view destination
 * @param [in]  src         Pointer to type erased device array view source
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_type_erased_device_array_view_copy(const hipgraph_resource_handle_t*         handle,
                                                hipgraph_type_erased_device_array_view_t* dst,
                                                const hipgraph_type_erased_device_array_view_t* src,
                                                hipgraph_error_t** error);

#ifdef __cplusplus
}
#endif
