# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
from hipgraph.layout import force_atlas2_wrapper
from hipgraph.utilities import ensure_hipgraph_obj_for_nx


def force_atlas2(
    input_graph,
    max_iter=500,
    pos_list=None,
    outbound_attraction_distribution=True,
    lin_log_mode=False,
    prevent_overlapping=False,
    edge_weight_influence=1.0,
    jitter_tolerance=1.0,
    barnes_hut_optimize=True,
    barnes_hut_theta=0.5,
    scaling_ratio=2.0,
    strong_gravity_mode=False,
    gravity=1.0,
    verbose=False,
    callback=None,
):
    """
    ForceAtlas2 is a continuous graph layout algorithm for handy network
    visualization.

    NOTE: Peak memory allocation occurs at 30*V.

    Parameters
    ----------
    input_graph : hipgraph.Graph
        hipGRAPH graph descriptor with connectivity information.
        Edge weights, if present, should be single or double precision
        floating point values.

    max_iter : integer, optional (default=500)
        This controls the maximum number of levels/iterations of the Force
        Atlas algorithm. When specified the algorithm will terminate after
        no more than the specified number of iterations.
        No error occurs when the algorithm terminates in this manner.
        Good short-term quality can be achieved with 50-100 iterations.
        Above 1000 iterations is discouraged.

    pos_list: cudf.DataFrame, optional (default=None)
        Data frame with initial vertex positions containing three columns:
        'vertex', 'x' and 'y' positions.

    outbound_attraction_distribution: bool, optional (default=True)
        Distributes attraction along outbound edges.
        Hubs attract less and thus are pushed to the borders.

    lin_log_mode: bool, optional (default=False)
        Switch Force Atlas model from lin-lin to lin-log.
        Makes clusters more tight.

    prevent_overlapping: bool, optional (default=False)
        Prevent nodes to overlap.

    edge_weight_influence: float, optional (default=1.0)
        How much influence you give to the edges weight.
        0 is “no influence” and 1 is “normal”.

    jitter_tolerance: float, optional (default=1.0)
        How much swinging you allow. Above 1 discouraged.
        Lower gives less speed and more precision.

    barnes_hut_optimize: bool, optional (default=True)
        Whether to use the Barnes Hut approximation or the slower
        exact version.

    barnes_hut_theta: float, optional (default=0.5)
        Float between 0 and 1. Tradeoff for speed (1) vs
        accuracy (0) for Barnes Hut only.

    scaling_ratio: float, optional (default=2.0)
        How much repulsion you want. More makes a more sparse graph.
        Switching from regular mode to LinLog mode needs a readjustment
        of the scaling parameter.

    strong_gravity_mode: bool, optional (default=False)
        Sets a force that attracts the nodes that are distant from the
        center more. It is so strong that it can sometimes dominate other
        forces.

    gravity : float, optional (default=1.0)
        Attracts nodes to the center. Prevents islands from drifting away.

    verbose: bool, optional (default=False)
        Output convergence info at each interation.

    callback: GraphBasedDimRedCallback, optional (default=None)
        An instance of GraphBasedDimRedCallback class to intercept
        the internal state of positions while they are being trained.

        Example of callback usage:
            from hipgraph.internals import GraphBasedDimRedCallback
                class CustomCallback(GraphBasedDimRedCallback):
                    def on_preprocess_end(self, positions):
                        print(positions.copy_to_host())
                    def on_epoch_end(self, positions):
                        print(positions.copy_to_host())
                    def on_train_end(self, positions):
                        print(positions.copy_to_host())

    Returns
    -------
    pos : cudf.DataFrame
        GPU data frame of size V containing three columns:
        the vertex identifiers and the x and y positions.

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> pos = hipgraph.force_atlas2(G)

    """
    input_graph, isNx = ensure_hipgraph_obj_for_nx(input_graph)

    if pos_list is not None:
        if not isinstance(pos_list, cudf.DataFrame):
            raise TypeError("pos_list should be a cudf.DataFrame")
        if set(pos_list.columns) != set(["x", "y", "vertex"]):
            raise ValueError("pos_list has wrong column names")
        if input_graph.renumbered is True:
            if input_graph.vertex_column_size() > 1:
                cols = pos_list.columns[:-2].to_list()
            else:
                cols = "vertex"
            pos_list = input_graph.add_internal_vertex_id(pos_list, "vertex", cols)

    if prevent_overlapping:
        raise Exception("Feature not supported")

    if input_graph.is_directed():
        input_graph = input_graph.to_undirected()

    pos = force_atlas2_wrapper.force_atlas2(
        input_graph,
        max_iter=max_iter,
        pos_list=pos_list,
        outbound_attraction_distribution=outbound_attraction_distribution,
        lin_log_mode=lin_log_mode,
        prevent_overlapping=prevent_overlapping,
        edge_weight_influence=edge_weight_influence,
        jitter_tolerance=jitter_tolerance,
        barnes_hut_optimize=barnes_hut_optimize,
        barnes_hut_theta=barnes_hut_theta,
        scaling_ratio=scaling_ratio,
        strong_gravity_mode=strong_gravity_mode,
        gravity=gravity,
        verbose=verbose,
        callback=callback,
    )

    if input_graph.renumbered:
        pos = input_graph.unrenumber(pos, "vertex")

    return pos
