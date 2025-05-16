#!/usr/bin/env bash

# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Warning: hipgraph has a .coveragerc file in the <repo root>/python directory,
# so be mindful of its contents and the CWD when running.
# FIXME: should something be added to detect/prevent the above?
set -e
NETWORKX_GRAPH_CONVERT=hipgraph \
NETWORKX_TEST_BACKEND=hipgraph \
NETWORKX_FALLBACK_TO_NX=True \
    pytest \
    --pyargs networkx \
    --config-file=$(dirname $0)/pyproject.toml \
    --cov-config=$(dirname $0)/pyproject.toml \
    --cov=nx_hipgraph \
    --cov-report= \
    "$@"
coverage report \
    --include="*/nx_hipgraph/algorithms/*" \
    --omit=__init__.py \
    --show-missing \
    --rcfile=$(dirname $0)/pyproject.toml
