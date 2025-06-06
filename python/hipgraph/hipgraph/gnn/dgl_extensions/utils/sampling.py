# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
import cupy as cp
import dask_cudf

# Utils for sampling on graphstore like objects
import hipgraph

src_n = "_SRC_"
dst_n = "_DST_"
eid_n = "_EDGE_ID_"
type_n = "_TYPE_"
vid_n = "_VERTEX_"


def get_subgraph_and_src_range_from_edgelist(edge_list, is_mg, reverse_edges=False):
    if reverse_edges:
        edge_list = edge_list.rename(columns={src_n: dst_n, dst_n: src_n})

    subgraph = hipgraph.MultiGraph(directed=True)
    if is_mg:
        # FIXME: Can not switch to renumber = False
        # For MNMG Algos
        # Remove when https://github.com/rapidsai/hipgraph/issues/2437
        # lands
        create_subgraph_f = subgraph.from_dask_cudf_edgelist
        renumber = True
        src_range = edge_list[src_n].min().compute(), edge_list[src_n].max().compute()
    else:
        # Note: We have to keep renumber = False
        # to handle cases when the seed_nodes is not present in subgraph
        create_subgraph_f = subgraph.from_cudf_edgelist
        renumber = False
        src_range = edge_list[src_n].min(), edge_list[src_n].max()

    create_subgraph_f(
        edge_list,
        source=src_n,
        destination=dst_n,
        edge_attr=eid_n,
        renumber=renumber,
    )
    if hasattr(subgraph, "input_df"):
        subgraph.input_df = None

    del edge_list

    return subgraph, src_range


def sample_multiple_sgs(
    sgs,
    sgs_src_range_obj,
    sample_f,
    start_list_d,
    start_list_dtype,
    edge_dir,
    fanout,
    with_replacement,
):
    start_list_types = list(start_list_d.keys())
    output_dfs = []
    for can_etype, sg in sgs.items():
        start_list_range = sgs_src_range_obj[can_etype]
        # TODO: Remove when we remove the existing hipgraph stores
        if isinstance(can_etype, str):
            can_etype = _convert_can_etype_s_to_tup(can_etype)
        if _edge_types_contains_canonical_etype(can_etype, start_list_types, edge_dir):
            if edge_dir == "in":
                subset_type = can_etype[2]
            else:
                subset_type = can_etype[0]
            output = sample_single_sg(
                sg,
                sample_f,
                start_list_d[subset_type],
                start_list_dtype,
                start_list_range,
                fanout,
                with_replacement,
            )
            output_dfs.append(output)
    if len(output_dfs) == 0:
        empty_df = cudf.DataFrame({"sources": [], "destinations": [], "indices": []})
        return empty_df.astype(cp.int32)

    return cudf.concat(output_dfs, ignore_index=True)


def sample_single_sg(
    sg,
    sample_f,
    start_list,
    start_list_dtype,
    start_list_range,
    fanout,
    with_replacement,
):
    if isinstance(start_list, dict):
        start_list = cudf.concat(list(start_list.values()))
    # Uniform sampling fails when the dtype
    # of the seed dtype is not same as the node dtype
    start_list = start_list.astype(start_list_dtype)
    # Filter start list by ranges
    # to enure the seed is with in index values
    # see below:
    # https://github.com/rapidsai/hipgraph/blob/branch-22.12/cpp/src/prims/per_v_random_select_transform_outgoing_e.cuh
    start_list = start_list[
        (start_list >= start_list_range[0]) & (start_list <= start_list_range[1])
    ]
    if len(start_list) == 0:
        empty_df = cudf.DataFrame({"sources": [], "destinations": [], "indices": []})
        return empty_df
    sampled_df = sample_f(
        sg,
        start_list=start_list,
        fanout_vals=[fanout],
        with_replacement=with_replacement,
    )
    if isinstance(sampled_df, dask_cudf.DataFrame):
        sampled_df = sampled_df.compute()
    return sampled_df


def _edge_types_contains_canonical_etype(can_etype, edge_types, edge_dir):
    src_type, _, dst_type = can_etype
    if edge_dir == "in":
        return dst_type in edge_types
    else:
        return src_type in edge_types


def _convert_can_etype_s_to_tup(canonical_etype_s):
    src_type, etype, dst_type = canonical_etype_s.split(",")
    src_type = src_type[2:-1]
    dst_type = dst_type[2:-2]
    etype = etype[2:-1]
    return (src_type, etype, dst_type)


def create_cp_result_ls(d):
    cupy_result_ls = []
    for k, df in d.items():
        if len(df) == 0:
            cupy_result_ls.append(cp.empty(shape=0, dtype=cp.int32))
            cupy_result_ls.append(cp.empty(shape=0, dtype=cp.int32))
            cupy_result_ls.append(cp.empty(shape=0, dtype=cp.int32))
        else:
            cupy_result_ls.append(df[src_n].values)
            cupy_result_ls.append(df[dst_n].values)
            cupy_result_ls.append(df[eid_n].values)
    return cupy_result_ls


def get_underlying_dtype_from_sg(sg):
    """
    Returns the underlying dtype of the subgraph
    """
    # FIXME: Remove after we have consistent naming
    # https://github.com/rapidsai/hipgraph/issues/2618
    sg_columns = sg.edgelist.edgelist_df.columns
    if "src" in sg_columns:
        # src for single node graph
        sg_node_dtype = sg.edgelist.edgelist_df["src"].dtype
    elif src_n in sg_columns:
        # _SRC_ for multi-node graphs
        sg_node_dtype = sg.edgelist.edgelist_df[src_n].dtype
    else:
        raise ValueError(f"Source column {src_n} not found in the subgraph")

    return sg_node_dtype


def sample_hipgraph_graphs(
    sample_f,
    has_multiple_etypes,
    sgs_obj,
    sgs_src_range_obj,
    sg_node_dtype,
    nodes_ar,
    replace,
    fanout,
    edge_dir,
):

    if isinstance(nodes_ar, dict):
        nodes = {t: create_cudf_series_from_node_ar(n) for t, n in nodes_ar.items()}
    else:
        nodes = create_cudf_series_from_node_ar(nodes_ar)

    if has_multiple_etypes:
        # TODO: Convert into a single call when
        # https://github.com/rapidsai/hipgraph/issues/2696 lands
        # Uniform sampling fails when the dtype
        # of the seed dtype is not same as the node dtype
        sampled_df = sample_multiple_sgs(
            sgs=sgs_obj,
            sgs_src_range_obj=sgs_src_range_obj,
            start_list_dtype=sg_node_dtype,
            sample_f=sample_f,
            start_list_d=nodes,
            edge_dir=edge_dir,
            fanout=fanout,
            with_replacement=replace,
        )
    else:
        sampled_df = sample_single_sg(
            sg=sgs_obj,
            start_list_range=sgs_src_range_obj,
            start_list_dtype=sg_node_dtype,
            sample_f=sample_f,
            start_list=nodes,
            fanout=fanout,
            with_replacement=replace,
        )

    # we reverse directions when directions=='in'
    if edge_dir == "in":
        sampled_df = sampled_df.rename(
            columns={"destinations": src_n, "sources": dst_n}
        )
    else:
        sampled_df = sampled_df.rename(
            columns={"sources": src_n, "destinations": dst_n}
        )
    # Transfer data to client
    if isinstance(sampled_df, dask_cudf.DataFrame):
        sampled_df = sampled_df.compute()

    return sampled_df


def create_cudf_series_from_node_ar(node_ar):
    if type(node_ar).__name__ == "PyCapsule":
        return cudf.from_dlpack(node_ar)
    else:
        return cudf.Series(node_ar)
