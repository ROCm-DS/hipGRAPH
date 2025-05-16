<!--
SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# `pylibhipgraph`

This directory contains the sources to the `pylibhipgraph` package. The sources
are primarily cython files which are built using the `setup.py` file in the
parent directory and depend on the `libhipgraph_c` and `libhipgraph` libraries and
headers.

## components
The `connected_components` APIs.

## structure
Internal utilities and types for use with the libhipgraph C++ library.

## utilities
Utility functions.

## experimental
This subpackage defines the "experimental" APIs. many of these APIs are defined
elsewhere and simply imported into the `experimental/__init__.py` file.

## tests
pytest tests for `pylibhipgraph`.
