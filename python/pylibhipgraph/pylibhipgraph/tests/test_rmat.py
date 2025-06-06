# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cupy as cp
import pytest
from pylibhipgraph import ResourceHandle, generate_rmat_edgelist

# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================


def check_results(
    result, scale, num_edges, include_edge_ids, include_edge_weights, include_edge_types
):

    h_src_arr, h_dst_arr, h_wgt_arr, h_ids_arr, h_types_arr = result

    if include_edge_weights:
        assert h_wgt_arr is not None
    if include_edge_ids:
        assert h_ids_arr is not None
    if include_edge_types:
        assert h_types_arr is not None

    vertices = cp.union1d(h_src_arr, h_dst_arr)
    assert len(h_src_arr) == len(h_dst_arr) == num_edges
    assert len(vertices) <= 2**scale


# TODO: Coverage for the MG implementation
@pytest.mark.parametrize("scale", [2, 4, 8])
@pytest.mark.parametrize("num_edges", [4, 16, 32])
@pytest.mark.parametrize("clip_and_flip", [False, True])
@pytest.mark.parametrize("scramble_vertex_ids", [False, True])
@pytest.mark.parametrize("include_edge_weights", [False, True])
@pytest.mark.parametrize("include_edge_types", [False, True])
@pytest.mark.parametrize("include_edge_ids", [False, True])
def test_rmat(
    scale,
    num_edges,
    clip_and_flip,
    scramble_vertex_ids,
    include_edge_weights,
    include_edge_types,
    include_edge_ids,
):

    resource_handle = ResourceHandle()

    result = generate_rmat_edgelist(
        resource_handle=resource_handle,
        random_state=42,
        scale=scale,
        num_edges=num_edges,
        a=0.57,
        b=0.19,
        c=0.19,
        clip_and_flip=clip_and_flip,
        scramble_vertex_ids=scramble_vertex_ids,
        include_edge_weights=include_edge_weights,
        minimum_weight=0,
        maximum_weight=1,
        dtype=cp.float32,
        include_edge_ids=include_edge_ids,
        include_edge_types=include_edge_types,
        min_edge_type_value=2,
        max_edge_type_value=5,
        multi_gpu=False,
    )
    check_results(
        result,
        scale,
        num_edges,
        include_edge_ids,
        include_edge_weights,
        include_edge_types,
    )
