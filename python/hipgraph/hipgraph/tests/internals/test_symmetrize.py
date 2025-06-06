# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import cudf
import hipgraph
import pandas as pd
import pytest
from hipgraph.testing import DEFAULT_DATASETS


@pytest.mark.sg
def test_version():
    gc.collect()
    hipgraph.__version__


def compare(src1, dst1, val1, src2, dst2, val2):
    #
    #  We will do comparison computations by using dataframe
    #  merge functions (essentially doing fast joins).  We
    #  start by making two data frames
    #
    df1 = cudf.DataFrame()
    df1["src1"] = src1
    df1["dst1"] = dst1
    if val1 is not None:
        df1["val1"] = val1

    df2 = cudf.DataFrame()
    df2["src2"] = src2
    df2["dst2"] = dst2
    if val2 is not None:
        df2["val2"] = val2

    #
    #  Check to see if all pairs in the original data frame
    #  still exist in the new data frame.  If we join (merge)
    #  the data frames where (src1[i]=src2[i]) and (dst1[i]=dst2[i])
    #  then we should get exactly the same number of entries in
    #  the data frame if we did not lose any data.
    #
    join = df1.merge(df2, left_on=["src1", "dst1"], right_on=["src2", "dst2"])

    if len(df1) != len(join):
        join2 = df1.merge(
            df2, how="left", left_on=["src1", "dst1"], right_on=["src2", "dst2"]
        )
        pd.set_option("display.max_rows", 500)
        print("df1 = \n", df1.sort_values(["src1", "dst1"]))
        print("df2 = \n", df2.sort_values(["src2", "dst2"]))
        print(
            "join2 = \n",
            join2.sort_values(["src1", "dst1"])
            .to_pandas()
            .query("src2.isnull()", engine="python"),
        )

    assert len(df1) == len(join)

    if val1 is not None:
        #
        #  Check the values.  In this join, if val1 and val2 are
        #  the same then we are good.  If they are different then
        #  we need to check if the value is selected from the opposite
        #  direction, so we'll merge with the edges reversed and
        #  check to make sure that the values all match
        #
        diffs = join.query("val1 != val2")
        diffs_check = diffs.merge(
            df1, left_on=["src1", "dst1"], right_on=["dst1", "src1"]
        )
        query = diffs_check.query("val1_y != val2")
        if len(query) > 0:
            print("differences: ")
            print(query)
            assert 0 == len(query)

    #
    #  Now check the symmetrized edges are present.  If the original
    #  data contains (u,v) we want to make sure that (v,u) is present
    #  in the new data frame.
    #
    #  We can accomplish this by doing the join (merge) where
    #  (src1[i] = dst2[i]) and (dst1[i] = src2[i]), and verifying
    #  that we get exactly the same number of entries in the data frame.
    #
    join = df1.merge(df2, left_on=["src1", "dst1"], right_on=["dst2", "src2"])
    assert len(df1) == len(join)

    if val1 is not None:
        #
        #  Check the values.  In this join, if val1 and val2 are
        #  the same then we are good.  If they are different then
        #  we need to check if the value is selected from the opposite
        #  direction, so we'll merge with the edges reversed and
        #  check to make sure that the values all match
        #
        diffs = join.query("val1 != val2")
        diffs_check = diffs.merge(
            df1, left_on=["src2", "dst2"], right_on=["src1", "dst1"]
        )
        query = diffs_check.query("val1_y != val2")
        if len(query) > 0:
            print("differences: ")
            print(query)
            assert 0 == len(query)

    #
    #  Finally, let's check (in both directions) backwards.
    #  We want to make sure that no edges were created in
    #  the symmetrize logic that didn't already exist in one
    #  direction or the other.  This is a bit more complicated.
    #
    #  The complication here is that the original data could,
    #  for some edge (u,v) ALREADY contain the edge (v,u).  The
    #  symmetrized graph will not duplicate any edges, so the edge
    #  (u,v) will only be present once.  So we can't simply check
    #  counts of df2 joined with df1.
    #
    #  join1 will contain the join (merge) of df2 to df1 in the
    #        forward direction
    #  join2 will contain the join (merge) of df2 to df1 in the
    #        reverse direction
    #
    #  Finally, we'll do an outer join of join1 and join2, which
    #  will combine any (u,v)/(v,u) pairs that might exist into
    #  a joined row while keeping any (u,v) pairs that don't exist
    #  in both data frames as single rows.  This gives us a data frame
    #  with the same number of rows as the symmetrized data.
    #
    join1 = df2.merge(df1, left_on=["src2", "dst2"], right_on=["src1", "dst1"])
    join2 = df2.merge(df1, left_on=["src2", "dst2"], right_on=["dst1", "src1"])
    joinM = join1.merge(join2, how="outer", on=["src2", "dst2"])

    assert len(df2) == len(joinM)

    #
    #  Note, we don't need to check the reverse values... we checked
    #  them in both directions earlier.
    #


@pytest.mark.sg
@pytest.mark.skip("debugging")
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_symmetrize_unweighted(graph_file):
    gc.collect()

    cu_M = graph_file.get_edgelist()
    sym_sources, sym_destinations = hipgraph.symmetrize(cu_M["src"], cu_M["dst"])

    #
    #  Check to see if all pairs in sources/destinations exist in
    #  both directions
    #
    compare(
        cu_M["src"],
        cu_M["dst"],
        None,
        sym_sources,
        sym_destinations,
        None,
    )


@pytest.mark.sg
@pytest.mark.skip("debugging")
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_symmetrize_weighted(graph_file):
    gc.collect()
    cu_M = graph_file.get_edgelist()

    sym_src, sym_dst, sym_w = hipgraph.symmetrize(cu_M["src"], cu_M["dst"], cu_M["wgt"])

    compare(cu_M["src"], cu_M["dst"], cu_M["wgt"], sym_src, sym_dst, sym_w)
