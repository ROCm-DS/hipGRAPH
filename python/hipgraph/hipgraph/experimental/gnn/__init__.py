# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from hipgraph.gnn.data_loading import BulkSampler
from pylibhipgraph.utilities.api_tools import promoted_experimental_warning_wrapper

# Passing in the namespace name of this module to the *_wrapper functions
# allows them to bypass the expensive inspect.stack() lookup.
_ns_name = __name__

BulkSampler = promoted_experimental_warning_wrapper(BulkSampler, _ns_name)
