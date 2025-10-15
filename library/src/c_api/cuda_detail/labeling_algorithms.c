// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/error.h>
#include <hipgraph_c/error.h>
#include <cugraph_c/graph.h>
#include <hipgraph_c/graph.h>
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_labeling_result_get_vertices(hipgraph_labeling_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_labeling_result_get_vertices((cugraph_labeling_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_labeling_result_get_labels(hipgraph_labeling_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_labeling_result_get_labels((cugraph_labeling_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_labeling_result_free(hipgraph_labeling_result_t* result)
{
    cugraph_labeling_result_free((cugraph_labeling_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_weakly_connected_components(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, bool do_expensive_check, hipgraph_labeling_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_weakly_connected_components((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (bool_t)do_expensive_check, (cugraph_labeling_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_strongly_connected_components(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, bool do_expensive_check, hipgraph_labeling_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_strongly_connected_components((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (bool_t)do_expensive_check, (cugraph_labeling_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}




