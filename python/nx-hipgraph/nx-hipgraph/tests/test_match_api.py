# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import importlib
import inspect

import networkx as nx
import nx_hipgraph as nxcg
from nx_hipgraph.utils import networkx_algorithm
from packaging.version import parse

nxver = parse(nx.__version__)


def test_match_signature_and_names():
    """Simple test to ensure our signatures and basic module layout match networkx."""
    for name, func in vars(nxcg.interface.BackendInterface).items():
        if not isinstance(func, networkx_algorithm):
            continue

        # nx version >=3.2 uses utils.backends, version >=3.0,<3.2 uses classes.backends
        is_nx_30_or_31 = hasattr(nx.classes, "backends")
        nx_backends = nx.classes.backends if is_nx_30_or_31 else nx.utils.backends

        if is_nx_30_or_31 and name in {"louvain_communities"}:
            continue
        if name not in nx_backends._registered_algorithms:
            print(f"{name} not dispatched from networkx")
            continue
        dispatchable_func = nx_backends._registered_algorithms[name]
        # nx version >=3.2 uses orig_func, version >=3.0,<3.2 uses _orig_func
        if is_nx_30_or_31:
            orig_func = dispatchable_func._orig_func
        else:
            orig_func = dispatchable_func.orig_func

        # Matching signatures?
        orig_sig = inspect.signature(orig_func)
        func_sig = inspect.signature(func)
        if not func.extra_params:
            assert orig_sig == func_sig
        else:
            # Ignore extra parameters added to nx-hipgraph algorithm
            # The key of func.extra_params may be like "max_level : int, optional",
            # but we only want "max_level" here.
            extra_params = {name.split(" ")[0] for name in func.extra_params}
            assert orig_sig == func_sig.replace(
                parameters=[
                    p
                    for name, p in func_sig.parameters.items()
                    if name not in extra_params
                ]
            )
        if func.can_run is not nxcg.utils.decorators._default_can_run:
            assert func_sig == inspect.signature(func.can_run)
        if func.should_run is not nxcg.utils.decorators._default_should_run:
            assert func_sig == inspect.signature(func.should_run)

        # Matching function names?
        assert func.__name__ == dispatchable_func.__name__ == orig_func.__name__

        # Matching dispatch names?
        # nx version >=3.2 uses name, version >=3.0,<3.2 uses dispatchname
        if is_nx_30_or_31:
            dispatchname = dispatchable_func.dispatchname
        else:
            dispatchname = dispatchable_func.name
        assert func.name == dispatchname

        # Matching modules (i.e., where function defined)?
        assert (
            "networkx." + func.__module__.split(".", 1)[1]
            == dispatchable_func.__module__
            == orig_func.__module__
        )

        # Matching package layout (i.e., which modules have the function)?
        nxcg_path = func.__module__
        name = func.__name__
        while "." in nxcg_path:
            # This only walks up the module tree and does not check sibling modules
            nxcg_path, mod_name = nxcg_path.rsplit(".", 1)
            nx_path = nxcg_path.replace("nx_hipgraph", "networkx")
            nxcg_mod = importlib.import_module(nxcg_path)
            nx_mod = importlib.import_module(nx_path)
            # Is the function present in the current module?
            present_in_nxcg = hasattr(nxcg_mod, name)
            present_in_nx = hasattr(nx_mod, name)
            if present_in_nxcg is not present_in_nx:  # pragma: no cover (debug)
                if present_in_nxcg:
                    raise AssertionError(
                        f"{name} exists in {nxcg_path}, but not in {nx_path}"
                    )
                raise AssertionError(
                    f"{name} exists in {nx_path}, but not in {nxcg_path}"
                )
            # Is the nested module present in the current module?
            present_in_nxcg = hasattr(nxcg_mod, mod_name)
            present_in_nx = hasattr(nx_mod, mod_name)
            if present_in_nxcg is not present_in_nx:  # pragma: no cover (debug)
                if present_in_nxcg:
                    raise AssertionError(
                        f"{mod_name} exists in {nxcg_path}, but not in {nx_path}"
                    )
                raise AssertionError(
                    f"{mod_name} exists in {nx_path}, but not in {nxcg_path}"
                )
