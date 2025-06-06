# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cupy as cp
import numpy as np
import pytest
from pylibhipgraph import GraphProperties, ResourceHandle, SGGraph, katz_centrality
from pylibhipgraph.testing import utils

TOY = utils.RAPIDS_DATASET_ROOT_DIR_PATH / "toy_graph_undirected.csv"


# =============================================================================
# Test helpers
# =============================================================================
def _get_param_args(param_name, param_values):
    """
    Returns a tuple of (<param_name>, <pytest.param list>) which can be applied
    as the args to pytest.mark.parametrize(). The pytest.param list also
    contains param id string formed from the param name and values.
    """
    return (param_name, [pytest.param(v, id=f"{param_name}={v}") for v in param_values])


def _generic_katz_test(
    src_arr,
    dst_arr,
    wgt_arr,
    result_arr,
    num_vertices,
    num_edges,
    store_transposed,
    alpha,
    beta,
    epsilon,
    max_iterations,
):
    """
    Builds a graph from the input arrays and runs katz using the other args,
    similar to how katz is tested in libhipgraph.
    """
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
    G = SGGraph(
        resource_handle=resource_handle,
        graph_properties=graph_props,
        src_or_offset_array=src_arr,
        dst_or_index_array=dst_arr,
        weight_array=wgt_arr,
        store_transposed=False,
        renumber=False,
        do_expensive_check=True,
    )

    (vertices, centralities) = katz_centrality(
        resource_handle,
        G,
        None,
        alpha,
        beta,
        epsilon,
        max_iterations,
        do_expensive_check=False,
    )

    result_arr = result_arr.get()
    vertices = vertices.get()
    centralities = centralities.get()

    for idx in range(num_vertices):
        vertex_id = vertices[idx]
        expected_result = result_arr[vertex_id]
        actual_result = centralities[idx]
        if pytest.approx(expected_result, 1e-4) != actual_result:
            raise ValueError(
                f"Vertex {idx} has centrality {actual_result}"
                f", should have been {expected_result}"
            )


def test_katz():
    num_edges = 8
    num_vertices = 6
    graph_data = np.genfromtxt(TOY, delimiter=" ")
    src = cp.asarray(graph_data[:, 0], dtype=np.int32)
    dst = cp.asarray(graph_data[:, 1], dtype=np.int32)
    wgt = cp.asarray(graph_data[:, 2], dtype=np.float32)
    result = cp.asarray(
        [0.410614, 0.403211, 0.390689, 0.415175, 0.395125, 0.433226], dtype=np.float32
    )
    alpha = 0.01
    beta = 1.0
    epsilon = 0.000001
    max_iterations = 1000

    # Katz requires store_transposed to be True
    _generic_katz_test(
        src,
        dst,
        wgt,
        result,
        num_vertices,
        num_edges,
        True,
        alpha,
        beta,
        epsilon,
        max_iterations,
    )
