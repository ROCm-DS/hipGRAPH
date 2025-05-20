# Copyright (c) 2015, Graphistry, Inc.
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import datetime as dt

import cudf
import hipgraph
import pandas as pd
import pytest
from cudf.testing.testing import assert_frame_equal

simple_df = cudf.DataFrame.from_pandas(
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "a1": [1, 2, 3],
            "a2": ["red", "blue", "green"],
            "🙈": ["æski ēˈmōjē", "😋", "s"],
        }
    )
)

hyper_df = cudf.DataFrame.from_pandas(
    pd.DataFrame({"aa": [0, 1, 2], "bb": ["a", "b", "c"], "cc": ["b", "0", "1"]})
)


@pytest.mark.sg
def test_complex_df():
    complex_df = pd.DataFrame(
        {
            "src": [0, 1, 2, 3],
            "dst": [1, 2, 3, 0],
            "colors": [1, 1, 2, 2],
            "bool": [True, False, True, True],
            "char": ["a", "b", "c", "d"],
            "str": ["a", "b", "c", "d"],
            "ustr": ["a", "b", "c", "d"],
            "emoji": ["😋", "😋😋", "😋", "😋"],
            "int": [0, 1, 2, 3],
            "num": [0.5, 1.5, 2.5, 3.5],
            "date_str": [
                "2018-01-01 00:00:00",
                "2018-01-02 00:00:00",
                "2018-01-03 00:00:00",
                "2018-01-05 00:00:00",
            ],
            "date": [
                dt.datetime(2018, 1, 1),
                dt.datetime(2018, 1, 1),
                dt.datetime(2018, 1, 1),
                dt.datetime(2018, 1, 1),
            ],
            "time": [
                pd.Timestamp("2018-01-05"),
                pd.Timestamp("2018-01-05"),
                pd.Timestamp("2018-01-05"),
                pd.Timestamp("2018-01-05"),
            ],
        }
    )

    for c in complex_df.columns:
        try:
            complex_df[c + "_cat"] = complex_df[c].astype("category")
        except Exception:
            # lists aren't categorical
            # print('could not make categorical', c)
            pass

    complex_df = cudf.DataFrame.from_pandas(complex_df)

    hipgraph.hypergraph(complex_df)


@pytest.mark.sg
@pytest.mark.parametrize("categorical_metadata", [False, True])
def test_hyperedges(categorical_metadata):

    h = hipgraph.hypergraph(simple_df, categorical_metadata=categorical_metadata)

    assert len(h.keys()) == len(["entities", "nodes", "edges", "events", "graph"])

    edges = cudf.from_pandas(
        pd.DataFrame(
            {
                "event_id": [
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                ],
                "edge_type": [
                    "a1",
                    "a1",
                    "a1",
                    "a2",
                    "a2",
                    "a2",
                    "id",
                    "id",
                    "id",
                    "🙈",
                    "🙈",
                    "🙈",
                ],
                "attrib_id": [
                    "a1::1",
                    "a1::2",
                    "a1::3",
                    "a2::red",
                    "a2::blue",
                    "a2::green",
                    "id::a",
                    "id::b",
                    "id::c",
                    "🙈::æski ēˈmōjē",
                    "🙈::😋",
                    "🙈::s",
                ],
                "id": ["a", "b", "c"] * 4,
                "a1": [1, 2, 3] * 4,
                "a2": ["red", "blue", "green"] * 4,
                "🙈": ["æski ēˈmōjē", "😋", "s"] * 4,
            }
        )
    )

    if categorical_metadata:
        edges = edges.astype({"edge_type": "category"})

    # check_like ignores the order of columns as long as all correct ones are present
    assert_frame_equal(edges, h["edges"], check_dtype=False, check_like=True)
    for k, v in [("entities", 12), ("nodes", 15), ("edges", 12), ("events", 3)]:
        assert len(h[k]) == v


@pytest.mark.sg
def test_hyperedges_direct():

    h = hipgraph.hypergraph(hyper_df, direct=True)

    assert len(h["edges"]) == 9
    assert len(h["nodes"]) == 9


@pytest.mark.sg
def test_hyperedges_direct_categories():

    h = hipgraph.hypergraph(
        hyper_df,
        direct=True,
        categories={
            "aa": "N",
            "bb": "N",
            "cc": "N",
        },
    )

    assert len(h["edges"]) == 9
    assert len(h["nodes"]) == 6


@pytest.mark.sg
def test_hyperedges_direct_manual_shaping():

    h1 = hipgraph.hypergraph(
        hyper_df,
        direct=True,
        EDGES={"aa": ["cc"], "cc": ["cc"]},
    )
    assert len(h1["edges"]) == 6

    h2 = hipgraph.hypergraph(
        hyper_df,
        direct=True,
        EDGES={"aa": ["cc", "bb", "aa"], "cc": ["cc"]},
    )
    assert len(h2["edges"]) == 12


@pytest.mark.sg
@pytest.mark.parametrize("categorical_metadata", [False, True])
def test_drop_edge_attrs(categorical_metadata):

    h = hipgraph.hypergraph(
        simple_df,
        columns=["id", "a1", "🙈"],
        drop_edge_attrs=True,
        categorical_metadata=categorical_metadata,
    )

    assert len(h.keys()) == len(["entities", "nodes", "edges", "events", "graph"])

    edges = cudf.DataFrame.from_pandas(
        pd.DataFrame(
            {
                "event_id": [
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                ],
                "edge_type": ["a1", "a1", "a1", "id", "id", "id", "🙈", "🙈", "🙈"],
                "attrib_id": [
                    "a1::1",
                    "a1::2",
                    "a1::3",
                    "id::a",
                    "id::b",
                    "id::c",
                    "🙈::æski ēˈmōjē",
                    "🙈::😋",
                    "🙈::s",
                ],
            }
        )
    )

    if categorical_metadata:
        edges = edges.astype({"edge_type": "category"})

    # check_like ignores the order of columns as long as all correct ones are present
    assert_frame_equal(edges, h["edges"], check_dtype=False, check_like=True)

    for k, v in [("entities", 9), ("nodes", 12), ("edges", 9), ("events", 3)]:
        assert len(h[k]) == v


@pytest.mark.sg
@pytest.mark.parametrize("categorical_metadata", [False, True])
def test_drop_edge_attrs_direct(categorical_metadata):

    h = hipgraph.hypergraph(
        simple_df,
        ["id", "a1", "🙈"],
        direct=True,
        drop_edge_attrs=True,
        EDGES={"id": ["a1"], "a1": ["🙈"]},
        categorical_metadata=categorical_metadata,
    )

    assert len(h.keys()) == len(["entities", "nodes", "edges", "events", "graph"])

    edges = cudf.DataFrame.from_pandas(
        pd.DataFrame(
            {
                "event_id": [
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                ],
                "edge_type": [
                    "a1::🙈",
                    "a1::🙈",
                    "a1::🙈",
                    "id::a1",
                    "id::a1",
                    "id::a1",
                ],
                "src": ["a1::1", "a1::2", "a1::3", "id::a", "id::b", "id::c"],
                "dst": [
                    "🙈::æski ēˈmōjē",
                    "🙈::😋",
                    "🙈::s",
                    "a1::1",
                    "a1::2",
                    "a1::3",
                ],
            }
        )
    )

    if categorical_metadata:
        edges = edges.astype({"edge_type": "category"})

    # check_like ignores the order of columns as long as all correct ones are present
    assert_frame_equal(edges, h["edges"], check_dtype=False, check_like=True)

    for k, v in [("entities", 9), ("nodes", 9), ("edges", 6), ("events", 0)]:
        assert len(h[k]) == v


@pytest.mark.sg
def test_skip_hyper():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "b"], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    )

    hg = hipgraph.hypergraph(df, SKIP=["c"], dropna=False)

    assert len(hg["graph"].nodes()) == 9
    assert len(hg["graph"].edges()) == 6


@pytest.mark.sg
def test_skip_drop_na_hyper():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "b"], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    )

    hg = hipgraph.hypergraph(df, SKIP=["c"], dropna=True)

    assert len(hg["graph"].nodes()) == 8
    assert len(hg["graph"].edges()) == 5


@pytest.mark.sg
def test_skip_direct():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "b"], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    )

    hg = hipgraph.hypergraph(df, SKIP=["c"], dropna=False, direct=True)

    assert len(hg["graph"].nodes()) == 6
    assert len(hg["graph"].edges()) == 3


@pytest.mark.sg
def test_skip_drop_na_direct():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "b"], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    )

    hg = hipgraph.hypergraph(df, SKIP=["c"], dropna=True, direct=True)

    assert len(hg["graph"].nodes()) == 4
    assert len(hg["graph"].edges()) == 2


@pytest.mark.sg
def test_drop_na_hyper():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "c"], "i": [1, 2, None]})
    )

    hg = hipgraph.hypergraph(df, dropna=True)

    assert len(hg["graph"].nodes()) == 7
    assert len(hg["graph"].edges()) == 4


@pytest.mark.sg
def test_drop_na_direct():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "a"], "i": [1, 1, None]})
    )

    hg = hipgraph.hypergraph(df, dropna=True, direct=True)

    assert len(hg["graph"].nodes()) == 2
    assert len(hg["graph"].edges()) == 1


@pytest.mark.sg
def test_skip_na_hyperedge():

    nans_df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"x": ["a", "b", "c"], "y": ["aa", None, "cc"]})
    )

    expected_hits = ["a", "b", "c", "aa", "cc"]

    skip_attr_h_edges = hipgraph.hypergraph(nans_df, drop_edge_attrs=True)["edges"]

    assert len(skip_attr_h_edges) == len(expected_hits)

    default_h_edges = hipgraph.hypergraph(nans_df)["edges"]
    assert len(default_h_edges) == len(expected_hits)


@pytest.mark.sg
def test_hyper_to_pa_vanilla():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"x": ["a", "b", "c"], "y": ["d", "e", "f"]})
    )

    hg = hipgraph.hypergraph(df)
    nodes_arr = hg["graph"].nodes().to_arrow()
    assert len(nodes_arr) == 9
    edges_err = hg["graph"].edges().to_arrow()
    assert len(edges_err) == 6


@pytest.mark.sg
def test_hyper_to_pa_mixed():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
    )

    hg = hipgraph.hypergraph(df)
    nodes_arr = hg["graph"].nodes().to_arrow()
    assert len(nodes_arr) == 9
    edges_err = hg["graph"].edges().to_arrow()
    assert len(edges_err) == 6


@pytest.mark.sg
def test_hyper_to_pa_na():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"x": ["a", None, "c"], "y": [1, 2, None]})
    )

    hg = hipgraph.hypergraph(df, dropna=False)
    print(hg["graph"].nodes())
    nodes_arr = hg["graph"].nodes().to_arrow()
    assert len(hg["graph"].nodes()) == 9
    assert len(nodes_arr) == 9
    edges_err = hg["graph"].edges().to_arrow()
    assert len(hg["graph"].edges()) == 6
    assert len(edges_err) == 6


@pytest.mark.sg
def test_hyper_to_pa_all():
    hg = hipgraph.hypergraph(simple_df, ["id", "a1", "🙈"])
    nodes_arr = hg["graph"].nodes().to_arrow()
    assert len(hg["graph"].nodes()) == 12
    assert len(nodes_arr) == 12
    edges_err = hg["graph"].edges().to_arrow()
    assert len(hg["graph"].edges()) == 9
    assert len(edges_err) == 9


@pytest.mark.sg
def test_hyper_to_pa_all_direct():
    hg = hipgraph.hypergraph(simple_df, ["id", "a1", "🙈"], direct=True)
    nodes_arr = hg["graph"].nodes().to_arrow()
    assert len(hg["graph"].nodes()) == 9
    assert len(nodes_arr) == 9
    edges_err = hg["graph"].edges().to_arrow()
    assert len(hg["graph"].edges()) == 9
    assert len(edges_err) == 9
