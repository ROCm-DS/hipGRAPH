#!/usr/bin/env python

# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

if __name__ == "__main__":
    import argparse

    from nx_hipgraph.scripts import print_table, print_tree

    parser = argparse.ArgumentParser(
        parents=[
            print_table.get_argumentparser(add_help=False),
            print_tree.get_argumentparser(add_help=False),
        ],
        description="Print info about functions implemented by nx-hipgraph",
    )
    parser.add_argument("action", choices=["print_table", "print_tree"])
    args = parser.parse_args()
    if args.action == "print_table":
        print_table.main()
    else:
        print_tree.main(
            by=args.by,
            networkx_path=args.networkx_path,
            dispatch_name=args.dispatch_name or args.dispatch_name_always,
            version_added=args.version_added,
            plc=args.plc,
            dispatch_name_if_different=not args.dispatch_name_always,
        )
