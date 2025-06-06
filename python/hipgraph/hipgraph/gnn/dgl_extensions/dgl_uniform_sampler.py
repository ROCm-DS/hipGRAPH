# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from functools import cached_property

import cudf
import cupy as cp
import hipgraph
from hipgraph.gnn.dgl_extensions.utils.sampling import (
    dst_n,
    get_subgraph_and_src_range_from_edgelist,
    get_underlying_dtype_from_sg,
    sample_hipgraph_graphs,
    src_n,
)


class DGLUniformSampler:
    """
    class object to do uniform sampling
    """

    def __init__(self, edge_list_dict, etype_range_dict, etype_id_dict, single_gpu):
        self.edge_list_dict = edge_list_dict
        self.etype_id_dict = etype_id_dict
        self.etype_id_range_dict = {
            self.etype_id_dict[etype]: r for etype, r in etype_range_dict.items()
        }
        self.single_gpu = single_gpu

    def sample_neighbors(
        self, nodes_ar, fanout=-1, edge_dir="in", prob=None, replace=False
    ):
        """
        Sample neighboring edges of the given nodes and return the subgraph.

        Parameters
        ----------
        nodes_ar : cupy array of node ids or dict with key of node type
                   and value of node ids gto sample neighbors from.
        fanout : int
            The number of edges to be sampled for each node on each edge type.
            If -1 is given all the neighboring edges for each node on
            each edge type will be selected.
        edge_dir : str {"in" or "out"}
            Determines whether to sample inbound or outbound edges.
            Can take either in for inbound edges or out for outbound edges.
        prob : str
            Feature name used as the (unnormalized) probabilities associated
            with each neighboring edge of a node. Each feature must be a
            scalar. The features must be non-negative floats, and the sum of
            the features of inbound/outbound edges for every node must be
            positive (though they don't have to sum up to one). Otherwise,
            the result will be undefined. If not specified, sample uniformly.
        replace : bool
            If True, sample with replacement.
        Returns
        -------
        [src, dst, eids] or {etype1:[src, dst, eids],...,}
        """

        if prob is not None:
            raise NotImplementedError(
                "prob is not currently supported",
                " for sample_neighbors in hipGRAPHStorage",
            )

        if edge_dir not in ["in", "out"]:
            raise ValueError(
                f"edge_dir must be either 'in' or 'out' got {edge_dir} instead"
            )

        if self.has_multiple_etypes:
            # TODO: Convert into a single call when
            # https://github.com/rapidsai/hipgraph/issues/2696 lands
            if edge_dir == "in":
                sgs_obj, sgs_src_range_obj = self.extracted_reverse_subgraphs_per_type
            else:
                sgs_obj, sgs_src_range_obj = self.extracted_subgraphs_per_type
            first_sg = list(sgs_obj.values())[0]
        else:
            if edge_dir == "in":
                sgs_obj, sgs_src_range_obj = self.extracted_reverse_subgraph
            else:
                sgs_obj, sgs_src_range_obj = self.extracted_subgraph

            first_sg = sgs_obj
        # Uniform sampling fails when the dtype
        # of the seed dtype is not same as the node dtype
        self.set_sg_node_dtype(first_sg)

        if self.single_gpu:
            sample_f = hipgraph.uniform_neighbor_sample
        else:
            sample_f = hipgraph.dask.uniform_neighbor_sample

        sampled_df = sample_hipgraph_graphs(
            sample_f=sample_f,
            has_multiple_etypes=self.has_multiple_etypes,
            sgs_obj=sgs_obj,
            sgs_src_range_obj=sgs_src_range_obj,
            sg_node_dtype=self._sg_node_dtype,
            nodes_ar=nodes_ar,
            replace=replace,
            fanout=fanout,
            edge_dir=edge_dir,
        )

        if self.has_multiple_etypes:
            # Heterogeneous graph case
            # Add type information
            return self._get_edgeid_type_d(sampled_df)
        else:
            return (
                sampled_df[src_n].astype("float").values,
                sampled_df[dst_n].astype("float").values,
                sampled_df["indices"].astype("float").values,
            )

    def _get_edgeid_type_d(self, df):
        df["type"] = self._get_type_id_from_indices(
            df["indices"], self.etype_id_range_dict
        )
        result_d = {
            etype: df[df["type"] == etype_id]
            for etype, etype_id in self.etype_id_dict.items()
        }
        return {
            etype: (
                df[src_n].astype("float").values,
                df[dst_n].astype("float").values,
                df["indices"].astype("float").values,
            )
            for etype, df in result_d.items()
        }

    @staticmethod
    def _get_type_id_from_indices(indices, etype_id_range_dict):
        type_ser = cudf.Series(
            cp.full(shape=len(indices), fill_value=-1, dtype=cp.int32)
        )

        for etype_id, (start, stop) in etype_id_range_dict.items():
            range_types = (start <= indices) & (indices < stop)
            type_ser[range_types] = type_ser.dtype.type(etype_id)

        return type_ser

    @cached_property
    def extracted_subgraph(self):
        assert len(self.edge_list_dict) == 1
        edge_list = list(self.edge_list_dict.values())[0]
        return get_subgraph_and_src_range_from_edgelist(
            edge_list,
            is_mg=not (self.single_gpu),
            reverse_edges=False,
        )

    @cached_property
    def extracted_reverse_subgraph(self):
        assert len(self.edge_list_dict) == 1
        edge_list = list(self.edge_list_dict.values())[0]
        return get_subgraph_and_src_range_from_edgelist(
            edge_list, is_mg=not (self.single_gpu), reverse_edges=True
        )

    @cached_property
    def extracted_subgraphs_per_type(self):
        sg_d = {}
        sg_src_range_d = {}
        for etype, edge_list in self.edge_list_dict.items():
            (
                sg_d[etype],
                sg_src_range_d[etype],
            ) = get_subgraph_and_src_range_from_edgelist(
                edge_list, is_mg=not (self.single_gpu), reverse_edges=False
            )
        return sg_d, sg_src_range_d

    @cached_property
    def extracted_reverse_subgraphs_per_type(self):
        sg_d = {}
        sg_src_range_d = {}
        for etype, edge_list in self.edge_list_dict.items():
            (
                sg_d[etype],
                sg_src_range_d[etype],
            ) = get_subgraph_and_src_range_from_edgelist(
                edge_list, is_mg=not (self.single_gpu), reverse_edges=True
            )
        return sg_d, sg_src_range_d

    @cached_property
    def has_multiple_etypes(self):
        return len(self.edge_list_dict) > 1

    @cached_property
    def etypes(self):
        return list(self.edge_list_dict.keys())

    def set_sg_node_dtype(self, sg):
        if hasattr(self, "_sg_node_dtype"):
            return self._sg_node_dtype
        else:
            self._sg_node_dtype = get_underlying_dtype_from_sg(sg)
        return self._sg_node_dtype
