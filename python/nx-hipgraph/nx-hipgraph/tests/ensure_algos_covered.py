# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""Ensure that all functions wrapped by @networkx_algorithm were called.

This file is run by CI and should not normally be run manually.
"""
import inspect
import json
from pathlib import Path

from nx_hipgraph.interface import BackendInterface
from nx_hipgraph.utils import networkx_algorithm

with Path("coverage.json").open() as f:
    coverage = json.load(f)

filenames_to_executed_lines = {
    "nx_hipgraph/"
    + filename.rsplit("nx_hipgraph/", 1)[-1]: set(coverage_info["executed_lines"])
    for filename, coverage_info in coverage["files"].items()
}


def unwrap(func):
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def get_func_filename(func):
    return "nx_hipgraph" + inspect.getfile(unwrap(func)).rsplit("nx_hipgraph", 1)[-1]


def get_func_linenos(func):
    lines, lineno = inspect.getsourcelines(unwrap(func))
    for i, line in enumerate(lines, lineno):
        if ":\n" in line:
            return set(range(i + 1, lineno + len(lines)))
    raise RuntimeError(f"Could not determine line numbers for function {func}")


def has_any_coverage(func):
    return bool(
        filenames_to_executed_lines[get_func_filename(func)] & get_func_linenos(func)
    )


def main():
    no_coverage = set()
    for attr, func in vars(BackendInterface).items():
        if not isinstance(func, networkx_algorithm):
            continue
        if not has_any_coverage(func):
            no_coverage.add(attr)
    if no_coverage:
        msg = "The following algorithms have no coverage: " + ", ".join(
            sorted(no_coverage)
        )
        # Create a border of "!"
        msg = (
            "\n\n"
            + "!" * (len(msg) + 6)
            + "\n!! "
            + msg
            + " !!\n"
            + "!" * (len(msg) + 6)
            + "\n"
        )
        raise AssertionError(msg)
    print("\nSuccess: coverage determined all algorithms were called!\n")


if __name__ == "__main__":
    main()
