# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import importlib
import sys
from types import ModuleType

# Override the individual NetworkX objects loaded above with the hipgraph.nx
# compat equivalents. This means if an equivalent compat obj is not available,
# the standard NetworkX obj will be used.
#
# Each hipgraph obj should have the same module path as the
# NetworkX obj it isoverriding, and the submodules along the hierarchy should
# each import the same sub objects/modules as NetworkX does. For example,
# in NetworkX, "pagerank" is a function in
# "networkx/algorithms/link_analysis/pagerank_alg.py", and is
# directly imported in the namespaces "networkx.algorithms.link_analysis",
# "networkx.algorithms", and "networkx". Therefore, the hipgraph
# compat pagerank should be defined in a module of the same name and
# also be present in the same namespaces.
# Refer to the networkx __init__.py files when adding new overriding
# modules to ensure the same paths and used and namespaces are populated.
from hipgraph.experimental.compat.nx import algorithms
from hipgraph.experimental.compat.nx.algorithms import *
from hipgraph.experimental.compat.nx.algorithms import link_analysis
from hipgraph.experimental.compat.nx.algorithms.link_analysis import *

# Start by populating this namespace with the same contents as
# networkx/__init__.py
from networkx import *

# FIXME: only perform the NetworkX imports below if NetworkX is installed. If
# it's determined that NetworkX is required to use nx compat, then the contents
# of this entire namespace may have to be optional, or packaged separately with
# a hard dependency on NetworkX.


# Recursively import all of the NetworkX modules into equivalent submodules
# under this package. The above "from networkx import *" handles names in this
# namespace, but it will not create the equivalent networkx submodule
# hierarchy. For example, a user could expect to "import hipgraph.nx.drawing",
# which should simply redirect to "networkx.drawing".
#
# This can be accomplished by updating sys.modules with the import path and
# module object of each NetworkX submodule in the NetworkX package hierarchy,
# but only for module paths that have not been added yet (otherwise this would
# overwrite the overides above).
_visited = set()


def _import_submodules_recursively(obj, mod_path):
    # Since modules can freely import any other modules, immediately mark this
    # obj as visited so submodules that import it are not re-examined
    # infinitely.
    _visited.add(obj)
    for name in dir(obj):
        sub_obj = getattr(obj, name)

        if type(sub_obj) is ModuleType:
            sub_mod_path = f"{mod_path}.{name}"
            # Do not overwrite modules that are already present, such as those
            # intended to override which were imported separately above.
            if sub_mod_path not in sys.modules:
                sys.modules[sub_mod_path] = sub_obj
            if sub_obj not in _visited:
                _import_submodules_recursively(sub_obj, sub_mod_path)


_import_submodules_recursively(importlib.import_module("networkx"), __name__)

del _visited
del _import_submodules_recursively

# At this point, individual types that hipgraph.nx are overriding
# could be used to override the corresponding types *inside* the
# networkx modules imported above. For example, the networkx graph generators
# will still return networkx.Graph objects instead of hipgraph.nx.Graph
# objects (unless the user knows to pass a "create_using" arg, if available).
# For specific overrides, assignments could be made in the imported
# a networkx modules so hipgraph.nx types are used by default.
# NOTE: this has the side-effect of causing all networkx
# imports in this python process/interpreter to use the override (ie. the user
# won't be able to use the original networkx types,
# even from a networkx import)
