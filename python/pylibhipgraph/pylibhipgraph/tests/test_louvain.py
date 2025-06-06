# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
import cupy as cp
import numpy as np
from pylibhipgraph import GraphProperties, ResourceHandle, SGGraph, louvain


def check_results(d_vertices, d_clusters, modularity):
    expected_vertices = np.array([1, 2, 3, 0, 4, 5], dtype=np.int32)
    expected_clusters = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    expected_modularity = 0.125

    h_vertices = d_vertices.get()
    h_clusters = d_clusters.get()

    assert np.array_equal(expected_vertices, h_vertices)
    assert np.array_equal(expected_clusters, h_clusters)
    assert expected_modularity == modularity


# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================
def test_sg_louvain_cupy():
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
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    max_level = 100
    threshold = 0.0001
    resolution = 1.0

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

    vertices, clusters, modularity = louvain(
        resource_handle, sg, max_level, threshold, resolution, do_expensive_check=False
    )

    check_results(vertices, clusters, modularity)


def test_sg_louvain_cudf():
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
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    max_level = 100
    threshold = 0.0001
    resolution = 1.0

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

    vertices, clusters, modularity = louvain(
        resource_handle, sg, max_level, threshold, resolution, do_expensive_check=False
    )

    check_results(vertices, clusters, modularity)
