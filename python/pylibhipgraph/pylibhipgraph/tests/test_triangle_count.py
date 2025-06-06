# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
import cupy as cp
import numpy as np
import pytest
from pylibhipgraph import (  # MGGraph,
    GraphProperties,
    ResourceHandle,
    SGGraph,
    triangle_count,
)


def check_results(d_result):
    expected_vertex_result = np.array([1, 2, 3, 0, 4, 5], dtype=np.int32)
    expected_counts_result = np.array([2, 2, 1, 1, 0, 0], dtype=np.int32)

    d_vertex_result, d_counts_result = d_result
    h_vertex_result = d_vertex_result.get()
    h_counts_result = d_counts_result.get()

    assert np.array_equal(expected_vertex_result, h_vertex_result)
    assert np.array_equal(expected_counts_result, h_counts_result)


# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================
def test_sg_triangle_count_cupy():
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=True, is_multigraph=False)

    device_srcs = cp.asarray(
        [0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32
    )
    device_dsts = cp.asarray(
        [1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32
    )
    device_weights = cp.asarray(
        [
            0.1,
            2.1,
            1.1,
            5.1,
            3.1,
            4.1,
            7.2,
            3.2,
            0.1,
            2.1,
            1.1,
            5.1,
            3.1,
            4.1,
            7.2,
            3.2,
        ],
        dtype=np.float32,
    )

    # FIXME: Disable the start_list parameter until it is working
    start_list = None

    sg = SGGraph(
        resource_handle=resource_handle,
        graph_properties=graph_props,
        src_or_offset_array=device_srcs,
        dst_or_index_array=device_dsts,
        weight_array=device_weights,
        store_transposed=False,
        renumber=True,
        do_expensive_check=False,
    )

    d_result = triangle_count(resource_handle, sg, start_list, do_expensive_check=True)

    check_results(d_result)


def test_sg_triangle_count_cudf():
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=True, is_multigraph=False)

    device_srcs = cudf.Series(
        [0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32
    )
    device_dsts = cudf.Series(
        [1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32
    )
    device_weights = cudf.Series(
        [
            0.1,
            2.1,
            1.1,
            5.1,
            3.1,
            4.1,
            7.2,
            3.2,
            0.1,
            2.1,
            1.1,
            5.1,
            3.1,
            4.1,
            7.2,
            3.2,
        ],
        dtype=np.float32,
    )
    # FIXME: Disable the start_list parameter until it is working
    start_list = None

    sg = SGGraph(
        resource_handle=resource_handle,
        graph_properties=graph_props,
        src_or_offset_array=device_srcs,
        dst_or_index_array=device_dsts,
        weight_array=device_weights,
        store_transposed=False,
        renumber=True,
        do_expensive_check=False,
    )

    d_result = triangle_count(resource_handle, sg, start_list, do_expensive_check=True)

    check_results(d_result)


# @pytest.mark.skip(reason="pylibhipgraph MG test infra not complete")
# def test_mg_triangle_count():
#     resource_handle = ResourceHandle()
#     graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
#
#     device_srcs = cp.asarray(
#         [0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32
#     )
#     device_dsts = cp.asarray(
#         [1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32
#     )
#     device_weights = cp.asarray(
#         [
#             0.1,
#             2.1,
#             1.1,
#             5.1,
#             3.1,
#             4.1,
#             7.2,
#             3.2,
#             0.1,
#             2.1,
#             1.1,
#             5.1,
#             3.1,
#             4.1,
#             7.2,
#             3.2,
#         ],
#         dtype=np.float32,
#     )
#
#     # FIXME: Disable the start_list parameter until it is working
#     start_list = None
#
#     mg = MGGraph(
#         resource_handle,
#         graph_props,
#         device_srcs,
#         device_dsts,
#         device_weights,
#         store_transposed=True,
#         num_edges=16,
#         do_expensive_check=False,
#     )
#
#     d_result = triangle_count(resource_handle, mg, start_list, do_expensive_check=True)
#     print(d_result)
