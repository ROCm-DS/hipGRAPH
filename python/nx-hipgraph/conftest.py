# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT


def pytest_addoption(parser):
    parser.addoption(
        "--bench",
        action="store_true",
        default=False,
        help="Run benchmarks (sugar for --benchmark-enable) and skip other tests"
        " (to run both benchmarks AND tests, use --all)",
    )
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run benchmarks AND tests (unlike --bench, which only runs benchmarks)",
    )
