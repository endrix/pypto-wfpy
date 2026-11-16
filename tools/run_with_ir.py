#!/usr/bin/env python3
# coding: utf-8
"""Run a PyPTO kernel with per-pass IR dumping enabled.

`compile_debug_mode` has no environment-variable hook, so it has to be set from
Python before the kernel's `@pypto.frontend.jit` body is compiled. This wrapper
sets it and then executes the kernel file as if it were __main__, leaving the
kernel itself unmodified.

    python3 tools/run_with_ir.py <kernel.py> [--run_mode sim] [--passes P1,P2]

With no --passes, every pass dumps (compile_debug_mode=1). With --passes, only
the named pass identifiers dump (dump_pass_graph), which is much faster and
smaller when you already know which pass you care about.
"""

from __future__ import annotations

import argparse
import runpy
import sys

import pypto


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="path to the kernel .py file")
    parser.add_argument("--run_mode", choices=["npu", "sim"], default="sim")
    parser.add_argument(
        "--passes",
        default="",
        help="comma-separated pass identifiers to dump; empty means all passes",
    )
    args = parser.parse_args()

    if args.passes:
        pypto.set_debug_options(dump_pass_graph=[p.strip() for p in args.passes.split(",") if p.strip()])
    else:
        pypto.set_debug_options(compile_debug_mode=1)

    # Hand the kernel the argv it expects, then run it as __main__ so its
    # `if __name__ == "__main__"` block executes.
    sys.argv = [args.kernel, "--run_mode", args.run_mode]
    try:
        runpy.run_path(args.kernel, run_name="__main__")
    except SystemExit as exc:  # the kernel returns its own exit code
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
