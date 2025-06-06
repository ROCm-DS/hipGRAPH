# Copyright (c) 2021-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import dask_cudf
import hipgraph
import pytest
from hipgraph.dask.common.mg_utils import is_single_gpu
from hipgraph.generators import rmat
from hipgraph.testing.mg_utils import start_dask_client, stop_dask_client

##############################################################################
_cluster = None
_client = None
_is_single_gpu = is_single_gpu()
_visible_devices = None
_scale_values = [2, 4, 16]
_scale_test_ids = [f"scale={x}" for x in _scale_values]
_graph_types = [hipgraph.Graph, None, int]
_graph_test_ids = [
    f"create_using={getattr(x, '__name__', str(x))}" for x in _graph_types
]


def _call_rmat(scale, num_edges, create_using, mg=True):
    """
    Simplifies calling RMAT by requiring only specific args that are varied by
    these tests and hard-coding all others.
    """
    return rmat(
        scale=scale,
        num_edges=num_edges,
        a=0.57,  # from Graph500
        b=0.19,  # from Graph500
        c=0.19,  # from Graph500
        seed=24,
        clip_and_flip=False,
        scramble_vertex_ids=True,
        create_using=create_using,
        mg=mg,
    )


###############################################################################
def setup_module():
    global _cluster
    global _client
    global _visible_devices
    if not _is_single_gpu:
        (_client, _cluster) = start_dask_client()
        _visible_devices = _client.scheduler_info()["workers"]


def teardown_module():
    if not _is_single_gpu:
        stop_dask_client(_client, _cluster)


###############################################################################
@pytest.mark.mg
@pytest.mark.filterwarnings("ignore:make_current is deprecated:DeprecationWarning")
@pytest.mark.parametrize("scale", _scale_values, ids=_scale_test_ids)
def test_rmat_edgelist(scale):
    """
    Verifies that the edgelist returned by rmat() is valid based on inputs.
    """
    if _is_single_gpu:
        pytest.skip("skipping MG testing on Single GPU system")

    num_edges = (2**scale) * 4
    create_using = None  # Returns the edgelist from RMAT

    df = _call_rmat(scale, num_edges, create_using)
    assert df.npartitions == len(_visible_devices)

    df = df.compute()
    assert len(df) == num_edges


@pytest.mark.mg
@pytest.mark.filterwarnings("ignore:make_current is deprecated:DeprecationWarning")
@pytest.mark.parametrize("graph_type", _graph_types, ids=_graph_test_ids)
def test_rmat_return_type(graph_type):
    """
    Verifies that the return type returned by rmat() is valid (or the proper
    exception is raised) based on inputs.
    """
    if _is_single_gpu:
        pytest.skip("skipping MG testing on Single GPU system")

    scale = 2
    num_edges = (2**scale) * 4

    if graph_type not in [hipgraph.Graph, None]:
        with pytest.raises(TypeError):
            _call_rmat(scale, num_edges, graph_type)
    else:
        G_or_df = _call_rmat(scale, num_edges, graph_type)

        if graph_type is None:
            assert type(G_or_df) is dask_cudf.DataFrame
        else:
            assert type(G_or_df) is graph_type
