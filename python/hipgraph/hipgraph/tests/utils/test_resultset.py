# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import cudf
from hipgraph.datasets.dataset import get_download_dir, set_download_dir
from hipgraph.testing.resultset import default_resultset_download_dir, load_resultset

###############################################################################


def test_load_resultset():
    with TemporaryDirectory() as tmpd:

        set_download_dir(Path(tmpd))
        default_resultset_download_dir.path = Path(tmpd) / "tests" / "resultsets"
        default_resultset_download_dir.path.mkdir(parents=True, exist_ok=True)

        datasets_download_dir = get_download_dir()
        resultsets_download_dir = default_resultset_download_dir.path
        assert "tests" in os.listdir(datasets_download_dir)
        assert "resultsets.tar.gz" not in os.listdir(datasets_download_dir / "tests")
        assert "traversal_mappings.csv" not in os.listdir(resultsets_download_dir)

        load_resultset(
            "traversal", "https://data.rapids.ai/hipgraph/results/resultsets.tar.gz"
        )

        assert "resultsets.tar.gz" in os.listdir(datasets_download_dir / "tests")
        assert "traversal_mappings.csv" in os.listdir(resultsets_download_dir)


def test_verify_resultset_load():
    # This test is more detailed than test_load_resultset, where for each module,
    # we check that every single resultset file is included along with the
    # corresponding mapping file.
    with TemporaryDirectory() as tmpd:
        set_download_dir(Path(tmpd))
        default_resultset_download_dir.path = Path(tmpd) / "tests" / "resultsets"
        default_resultset_download_dir.path.mkdir(parents=True, exist_ok=True)

        resultsets_download_dir = default_resultset_download_dir.path

        load_resultset(
            "traversal", "https://data.rapids.ai/hipgraph/results/resultsets.tar.gz"
        )

        resultsets = os.listdir(resultsets_download_dir)
        downloaded_results = cudf.read_csv(
            resultsets_download_dir / "traversal_mappings.csv", sep=" "
        )
        downloaded_uuids = downloaded_results["#UUID"].values
        for resultset_uuid in downloaded_uuids:
            assert str(resultset_uuid) + ".csv" in resultsets
