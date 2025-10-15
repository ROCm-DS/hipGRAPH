// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/array.h>
#include <hipgraph_c/array.h>
#include <cugraph_c/coo.h>
#include <hipgraph_c/coo.h>
#include <cugraph_c/graph.h>
#include <hipgraph_c/graph.h>
#include <cugraph_c/random.h>
#include <hipgraph_c/random.h>
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_generate_rmat_edgelist(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, size_t scale, size_t num_edges, double a, double b, double c, bool clip_and_flip, bool scramble_vertex_ids, hipgraph_coo_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_generate_rmat_edgelist((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, scale, num_edges, a, b, c, (bool_t)clip_and_flip, (bool_t)scramble_vertex_ids, (cugraph_coo_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_generate_rmat_edgelists(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, size_t n_edgelists, size_t min_scale, size_t max_scale, size_t edge_factor, hipgraph_generator_distribution_t size_distribution, hipgraph_generator_distribution_t edge_distribution, bool clip_and_flip, bool scramble_vertex_ids, hipgraph_coo_list_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_generate_rmat_edgelists((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, n_edgelists, min_scale, max_scale, edge_factor, _hipgraph_to_cugraph_generator_distribution_t(size_distribution), _hipgraph_to_cugraph_generator_distribution_t(edge_distribution), (bool_t)clip_and_flip, (bool_t)scramble_vertex_ids, (cugraph_coo_list_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_generate_edge_weights(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_coo_t* coo, hipgraph_data_type_id_t dtype, double minimum_weight, double maximum_weight, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_generate_edge_weights((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_coo_t*)coo, _hipgraph_to_cugraph_data_type_id_t(dtype), minimum_weight, maximum_weight, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_generate_edge_ids(const hipgraph_resource_handle_t* handle, hipgraph_coo_t* coo, bool multi_gpu, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_generate_edge_ids((const cugraph_resource_handle_t*)handle, (cugraph_coo_t*)coo, (bool_t)multi_gpu, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_generate_edge_types(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_coo_t* coo, int32_t min_edge_type, int32_t max_edge_type, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_generate_edge_types((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_coo_t*)coo, min_edge_type, max_edge_type, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}




