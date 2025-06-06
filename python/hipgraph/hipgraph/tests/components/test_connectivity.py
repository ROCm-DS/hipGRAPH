# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import time
from collections import defaultdict

import cudf
import cupy as cp
import hipgraph
import networkx as nx
import numpy as np
import pytest
from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix
from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from hipgraph.datasets import dolphins, email_Eu_core, netscience
from hipgraph.testing import DEFAULT_DATASETS, utils
from hipgraph.utilities import is_nx_graph_type
from scipy.sparse import coo_matrix as sp_coo_matrix
from scipy.sparse import csc_matrix as sp_csc_matrix
from scipy.sparse import csr_matrix as sp_csr_matrix

DATASETS_BATCH = [dolphins, netscience, email_Eu_core]


print("Networkx version : {} ".format(nx.__version__))

# Map of hipGRAPH input types to the expected output type for hipGRAPH
# connected_components calls.
hipGRAPH_input_output_map = {
    hipgraph.Graph: cudf.DataFrame,
    nx.Graph: dict,
    nx.DiGraph: dict,
    cp_coo_matrix: tuple,
    cp_csr_matrix: tuple,
    cp_csc_matrix: tuple,
    sp_coo_matrix: tuple,
    sp_csr_matrix: tuple,
    sp_csc_matrix: tuple,
}
cupy_types = [cp_coo_matrix, cp_csr_matrix, cp_csc_matrix]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Helper functions
# =============================================================================
def networkx_weak_call(graph_file):
    G = graph_file.get_graph()
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    Gnx = nx.from_pandas_edgelist(M, source="0", target="1", create_using=nx.DiGraph())

    # Weakly Connected components call:
    t1 = time.time()
    result = nx.weakly_connected_components(Gnx)
    t2 = time.time() - t1
    print("Time : " + str(t2))

    nx_labels = sorted(result)
    nx_n_components = len(nx_labels)
    lst_nx_components = sorted(nx_labels, key=len, reverse=True)

    return (G, dataset_path, nx_labels, nx_n_components, lst_nx_components, "weak")


def networkx_strong_call(graph_file):
    G = graph_file.get_graph(create_using=hipgraph.Graph(directed=True))
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    Gnx = nx.from_pandas_edgelist(M, source="0", target="1", create_using=nx.DiGraph())

    t1 = time.time()
    result = nx.strongly_connected_components(Gnx)
    t2 = time.time() - t1
    print("Time : " + str(t2))

    nx_labels = sorted(result)
    nx_n_components = len(nx_labels)
    lst_nx_components = sorted(nx_labels, key=len, reverse=True)

    return (G, dataset_path, nx_labels, nx_n_components, lst_nx_components, "strong")


def hipgraph_call(
    gpu_benchmark_callable, hipgraph_algo, input_G_or_matrix, directed=None
):
    """
    Test helper that calls hipgraph_algo (which is either
    weakly_connected_components() or strongly_connected_components()) on the
    Graph or matrix object input_G_or_matrix, via the gpu_benchmark_callable
    benchmark callable (which may or may not perform benchmarking based on
    command-line options), verify the result type, and return a dictionary for
    comparison.
    """
    # if benchmarking is enabled, this call will be benchmarked (ie. run
    # repeatedly, run time averaged, etc.)
    result = gpu_benchmark_callable(hipgraph_algo, input_G_or_matrix, directed)

    # dict of labels to list of vertices with that label
    label_vertex_dict = defaultdict(list)

    # Lookup results differently based on return type, and ensure return type
    # is correctly set based on input type.
    input_type = type(input_G_or_matrix)
    expected_return_type = hipGRAPH_input_output_map[input_type]

    if expected_return_type is cudf.DataFrame:
        assert type(result) is cudf.DataFrame
        for i in range(len(result)):
            label_vertex_dict[result["labels"][i]].append(result["vertex"][i])

    # NetworkX input results in returning a dictionary mapping vertices to
    # their labels.
    elif expected_return_type is dict:
        assert type(result) is dict
        for vert, label in result.items():
            label_vertex_dict[label].append(vert)

    # A CuPy/SciPy input means the return value will be a 2-tuple of:
    #   n_components: int
    #       The number of connected components (number of unique labels).
    #   labels: ndarray
    #       The length-N array of labels of the connected components.
    elif expected_return_type is tuple:
        assert type(result) is tuple
        assert type(result[0]) is int
        if input_type in cupy_types:
            assert type(result[1]) is cp.ndarray
        else:
            assert type(result[1]) is np.ndarray

        unique_labels = set([n.item() for n in result[1]])
        assert len(unique_labels) == result[0]

        # The returned dict used in the tests for checking correctness needs
        # the actual vertex IDs, which are not in the returned data (the
        # CuPy/SciPy connected_components return types hipGRAPH is converting
        # to does not include them). So, extract the vertices from the input
        # COO, order them to match the returned list of labels (which is just
        # a sort), and include them in the returned dict.
        if input_type in [cp_csr_matrix, cp_csc_matrix, sp_csr_matrix, sp_csc_matrix]:
            coo = input_G_or_matrix.tocoo(copy=False)
        else:
            coo = input_G_or_matrix
        verts = sorted(set([n.item() for n in coo.col] + [n.item() for n in coo.row]))
        num_verts = len(verts)
        num_verts_assigned_labels = len(result[1])
        assert num_verts_assigned_labels == num_verts

        for i in range(num_verts):
            label = result[1][i].item()
            label_vertex_dict[label].append(verts[i])

    else:
        raise RuntimeError(f"unsupported return type: {expected_return_type}")

    return label_vertex_dict


def which_cluster_idx(_cluster, _find_vertex):
    idx = -1
    for i in range(len(_cluster)):
        if _find_vertex in _cluster[i]:
            idx = i
            break
    return idx


def assert_scipy_api_compat(G, dataset_path, api_type):
    """
    Ensure hipgraph.scc() and hipgraph.connected_components() can be used as
    drop-in replacements for scipy.connected_components():

    scipy.sparse.csgraph.connected_components(csgraph,
                                              directed=True,
                                              connection='weak',
                                              return_labels=True)
    Parameters
    ----------
        csgraph : array_like or sparse matrix
            The N x N matrix representing the compressed sparse graph. The
            input csgraph will be converted to csr format for the calculation.
        directed : bool, optional
            If True (default), then operate on a directed graph: only move from
            point i to point j along paths csgraph[i, j]. If False, then find
            the shortest path on an undirected graph: the algorithm can
            progress from point i to j along csgraph[i, j] or csgraph[j, i].
        connection : str, optional
            [‘weak’|’strong’]. For directed graphs, the type of connection to
            use. Nodes i and j are strongly connected if a path exists both
            from i to j and from j to i. A directed graph is weakly connected
            if replacing all of its directed edges with undirected edges
            produces a connected (undirected) graph. If directed == False, this
            keyword is not referenced.
        return_labels : bool, optional
            If True (default), then return the labels for each of the connected
            components.

    Returns
    -------
        n_components : int
            The number of connected components.
        labels : ndarray
            The length-N array of labels of the connected components.
    """
    api_call = {
        "strong": hipgraph.strongly_connected_components,
        "weak": hipgraph.weakly_connected_components,
    }[api_type]
    connection = api_type
    wrong_connection = {"strong": "weak", "weak": "strong"}[api_type]

    input_hipgraph_graph = G
    input_coo_matrix = utils.create_obj_from_csv(
        dataset_path, cp_coo_matrix, edgevals=True
    )

    # Ensure scipy-only options are rejected for hipgraph inputs
    with pytest.raises(TypeError):
        api_call(input_hipgraph_graph, directed=False)
    with pytest.raises(TypeError):
        api_call(input_hipgraph_graph, return_labels=False)

    # Setting connection to strong for strongly_* and weak for weakly_* is
    # redundant, but valid

    api_call(input_hipgraph_graph, connection=connection)

    # Invalid for the API
    with pytest.raises(TypeError):
        (n_components, labels) = api_call(
            input_coo_matrix, directed=False, connection=wrong_connection
        )

    (n_components, labels) = api_call(input_coo_matrix, directed=False)
    (n_components, labels) = api_call(
        input_coo_matrix, directed=False, connection=connection
    )
    n_components = api_call(input_coo_matrix, directed=False, return_labels=False)
    assert type(n_components) is int


# =============================================================================
# Pytest fixtures
# =============================================================================
@pytest.fixture(scope="module", params=DEFAULT_DATASETS)
def dataset_nxresults_weak(request):
    return networkx_weak_call(request.param)


@pytest.fixture(scope="module", params=[DEFAULT_DATASETS[0]])
def single_dataset_nxresults_weak(request):
    return networkx_weak_call(request.param)


@pytest.fixture(scope="module", params=DATASETS_BATCH)
def dataset_nxresults_strong(request):
    return networkx_strong_call(request.param)


@pytest.fixture(scope="module", params=[DATASETS_BATCH[0]])
def single_dataset_nxresults_strong(request):
    return networkx_strong_call(request.param)


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.sg
@pytest.mark.parametrize("hipgraph_input_type", utils.HIPGRAPH_DIR_INPUT_TYPES)
def test_weak_cc(gpubenchmark, dataset_nxresults_weak, hipgraph_input_type):
    (
        G,
        dataset_path,
        netx_labels,
        nx_n_components,
        lst_nx_components,
        api_type,
    ) = dataset_nxresults_weak

    # hipGRAPH or nx 'input_type' should have this parameter set to None
    directed = None
    if not isinstance(hipgraph_input_type, hipgraph.Graph):
        input_G_or_matrix = utils.create_obj_from_csv(
            dataset_path, hipgraph_input_type, edgevals=True
        )
        if not is_nx_graph_type(hipgraph_input_type):
            # directed should be set to False when creating a hipGRAPH from
            # neither a hipGRAPH nor nx type
            directed = False
    else:
        input_G_or_matrix = G
    hipgraph_labels = hipgraph_call(
        gpubenchmark, hipgraph.weakly_connected_components, input_G_or_matrix, directed
    )

    # while hipgraph returns a component label for each vertex;
    cg_n_components = len(hipgraph_labels)

    # Comapre number of components
    assert nx_n_components == cg_n_components

    lst_nx_components_lens = [len(c) for c in lst_nx_components]

    hipgraph_vertex_lst = hipgraph_labels.values()
    lst_cg_components = sorted(hipgraph_vertex_lst, key=len, reverse=True)
    lst_cg_components_lens = [len(c) for c in lst_cg_components]

    # Compare lengths of each component
    assert lst_nx_components_lens == lst_cg_components_lens

    # Compare vertices of largest component
    nx_vertices = sorted(lst_nx_components[0])
    first_vert = nx_vertices[0]

    idx = which_cluster_idx(lst_cg_components, first_vert)
    assert idx != -1, "Check for Nx vertex in hipGRAPH results failed"

    cg_vertices = sorted(lst_cg_components[idx])

    assert nx_vertices == cg_vertices


@pytest.mark.sg
@pytest.mark.parametrize(
    "hipgraph_input_type", utils.NX_DIR_INPUT_TYPES + utils.MATRIX_INPUT_TYPES
)
def test_weak_cc_nonnative_inputs(
    gpubenchmark, single_dataset_nxresults_weak, hipgraph_input_type
):
    test_weak_cc(gpubenchmark, single_dataset_nxresults_weak, hipgraph_input_type)


@pytest.mark.sg
@pytest.mark.parametrize("hipgraph_input_type", utils.HIPGRAPH_DIR_INPUT_TYPES)
def test_strong_cc(gpubenchmark, dataset_nxresults_strong, hipgraph_input_type):

    # NetX returns a list of components, each component being a
    # collection (set{}) of vertex indices
    (
        G,
        dataset_path,
        netx_labels,
        nx_n_components,
        lst_nx_components,
        api_type,
    ) = dataset_nxresults_strong

    if not isinstance(hipgraph_input_type, hipgraph.Graph):
        input_G_or_matrix = utils.create_obj_from_csv(
            dataset_path, hipgraph_input_type, edgevals=True
        )
    else:
        input_G_or_matrix = G
    hipgraph_labels = hipgraph_call(
        gpubenchmark, hipgraph.strongly_connected_components, input_G_or_matrix
    )

    if isinstance(hipgraph_input_type, hipgraph.Graph):
        assert isinstance(input_G_or_matrix, type(hipgraph_input_type))
    else:
        assert isinstance(input_G_or_matrix, hipgraph_input_type)
    # while hipgraph returns a component label for each vertex;
    cg_n_components = len(hipgraph_labels)

    # Comapre number of components found
    assert nx_n_components == cg_n_components

    lst_nx_components_lens = [len(c) for c in lst_nx_components]

    hipgraph_vertex_lst = hipgraph_labels.values()
    lst_cg_components = sorted(hipgraph_vertex_lst, key=len, reverse=True)
    lst_cg_components_lens = [len(c) for c in lst_cg_components]

    # Compare lengths of each component
    assert lst_nx_components_lens == lst_cg_components_lens

    # Compare vertices of largest component
    # note that there might be more than one largest component
    nx_vertices = sorted(lst_nx_components[0])
    first_vert = nx_vertices[0]

    idx = which_cluster_idx(lst_cg_components, first_vert)
    assert idx != -1, "Check for Nx vertex in hipGRAPH results failed"

    cg_vertices = sorted(lst_cg_components[idx])
    assert nx_vertices == cg_vertices


@pytest.mark.sg
@pytest.mark.parametrize(
    "hipgraph_input_type", utils.NX_DIR_INPUT_TYPES + utils.MATRIX_INPUT_TYPES
)
def test_strong_cc_nonnative_inputs(
    gpubenchmark, single_dataset_nxresults_strong, hipgraph_input_type
):
    test_strong_cc(gpubenchmark, single_dataset_nxresults_strong, hipgraph_input_type)


@pytest.mark.sg
def test_scipy_api_compat_weak(single_dataset_nxresults_weak):
    (G, dataset_path, _, _, _, api_type) = single_dataset_nxresults_weak
    assert_scipy_api_compat(G, dataset_path, api_type)


@pytest.mark.sg
def test_scipy_api_compat_strong(single_dataset_nxresults_strong):
    (G, dataset_path, _, _, _, api_type) = single_dataset_nxresults_strong
    assert_scipy_api_compat(G, dataset_path, api_type)


@pytest.mark.sg
@pytest.mark.parametrize("connection_type", ["strong", "weak"])
def test_scipy_api_compat(connection_type):
    if connection_type == "strong":
        graph_file = DATASETS_BATCH[0]
    else:
        graph_file = DEFAULT_DATASETS[0]

    input_hipgraph_graph = graph_file.get_graph()

    dataset_path = graph_file.get_path()

    input_coo_matrix = utils.create_obj_from_csv(
        dataset_path, cp_coo_matrix, edgevals=True
    )

    # connection is the only API that is accepted with hipgraph objs
    retval = hipgraph.connected_components(
        input_hipgraph_graph, connection=connection_type
    )
    assert type(retval) is cudf.DataFrame

    # Ensure scipy-only options (except connection) are rejected for hipgraph
    # inputs
    with pytest.raises(TypeError):
        hipgraph.connected_components(input_hipgraph_graph, directed=True)
    with pytest.raises(TypeError):
        hipgraph.connected_components(input_hipgraph_graph, return_labels=False)
    with pytest.raises(TypeError):
        hipgraph.connected_components(
            input_hipgraph_graph, connection=connection_type, return_labels=False
        )

    # only accept weak or strong
    with pytest.raises(ValueError):
        hipgraph.connected_components(input_hipgraph_graph, connection="invalid")

    (n_components, labels) = hipgraph.connected_components(
        input_coo_matrix, directed=False, connection=connection_type
    )
    # FIXME: connection should default to "weak", need to test that
    (n_components, labels) = hipgraph.connected_components(
        input_coo_matrix, directed=False
    )
    n_components = hipgraph.connected_components(
        input_coo_matrix, directed=False, return_labels=False
    )
    assert type(n_components) is int
