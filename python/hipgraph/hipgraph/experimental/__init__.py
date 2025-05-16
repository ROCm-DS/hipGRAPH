# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from pylibhipgraph.utilities.api_tools import (
    experimental_warning_wrapper,
    promoted_experimental_warning_wrapper,
)

# Passing in the namespace name of this module to the *_wrapper functions
# allows them to bypass the expensive inspect.stack() lookup.
_ns_name = __name__

from hipgraph.structure.property_graph import EXPERIMENTAL__PropertyGraph

PropertyGraph = experimental_warning_wrapper(EXPERIMENTAL__PropertyGraph, _ns_name)

from hipgraph.structure.property_graph import EXPERIMENTAL__PropertySelection

PropertySelection = experimental_warning_wrapper(
    EXPERIMENTAL__PropertySelection, _ns_name
)

# from hipgraph.dask.structure.mg_property_graph import EXPERIMENTAL__MGPropertyGraph
#
# MGPropertyGraph = experimental_warning_wrapper(EXPERIMENTAL__MGPropertyGraph, _ns_name)
#
# from hipgraph.dask.structure.mg_property_graph import EXPERIMENTAL__MGPropertySelection
#
# MGPropertySelection = experimental_warning_wrapper(
#     EXPERIMENTAL__MGPropertySelection, _ns_name
# )

from hipgraph.experimental.components.scc import (
    EXPERIMENTAL__strong_connected_component,
)

strong_connected_component = experimental_warning_wrapper(
    EXPERIMENTAL__strong_connected_component, _ns_name
)

# from hipgraph.gnn.data_loading import BulkSampler

# BulkSampler = promoted_experimental_warning_wrapper(BulkSampler, _ns_name)
