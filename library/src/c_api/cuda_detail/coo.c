// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/array.h>
#include <hipgraph_c/array.h>
#include <cugraph_c/graph.h>
#include <hipgraph_c/graph.h>
#include <cugraph_c/random.h>
#include <hipgraph_c/random.h>
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>

HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_sources(hipgraph_coo_t* coo)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_coo_get_sources((cugraph_coo_t*)coo);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_destinations(hipgraph_coo_t* coo)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_coo_get_destinations((cugraph_coo_t*)coo);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_weights(hipgraph_coo_t* coo)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_coo_get_edge_weights((cugraph_coo_t*)coo);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_id(hipgraph_coo_t* coo)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_coo_get_edge_id((cugraph_coo_t*)coo);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_type(hipgraph_coo_t* coo)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_coo_get_edge_type((cugraph_coo_t*)coo);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT size_t hipgraph_coo_list_size(const hipgraph_coo_list_t* coo_list)
{
    size_t out;
    out = cugraph_coo_list_size((const cugraph_coo_list_t*)coo_list);
    return (size_t)out;
}


HIPGRAPH_EXPORT hipgraph_coo_t* hipgraph_coo_list_element(hipgraph_coo_list_t* coo_list, size_t index)
{
    cugraph_coo_t* out;
    out = cugraph_coo_list_element((cugraph_coo_list_t*)coo_list, index);
    return (hipgraph_coo_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_coo_free(hipgraph_coo_t* coo)
{
    cugraph_coo_free((cugraph_coo_t*)coo);
}


HIPGRAPH_EXPORT void hipgraph_coo_list_free(hipgraph_coo_list_t* coo_list)
{
    cugraph_coo_list_free((cugraph_coo_list_t*)coo_list);
}




