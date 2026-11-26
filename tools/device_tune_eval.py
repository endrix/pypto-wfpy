"""Run a frozen adapter in an isolated directory with the project's CANN environment."""

import argparse
from contextlib import contextmanager, chdir
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def setup(adapter, path, mode, profile=False, module_name="candidate_kernel"):
    import torch
    import pypto
    from tools.check_kda import load_module

    device = torch.device("cpu")
    if mode == "npu":
        import torch_npu
        device = torch.device(f"npu:{os.environ['TILE_FWK_DEVICE_ID']}")
        torch_npu.npu.set_device(device)
    pypto.set_debug_options(runtime_debug_mode=int(profile))
    return import_module(adapter), load_module(module_name, path) if path is not None else None, device


@contextmanager
def jit_cache(cache):
    """Scope the installed PyPTO frontend's global KernelModule cache to one implementation.

    Factory-defined JIT bodies can share cache keys across modules when a change
    only touches a helper or module-level tile constant. The NPU launcher permits
    one owning PyPTO process per card, so isolate the caches in that process.
    """
    from pypto.frontend.parser.entry import JitCallableWrapper
    previous = JitCallableWrapper._kernel_module_cache
    JitCallableWrapper._kernel_module_cache = cache
    try:
        yield
    finally:
        JitCallableWrapper._kernel_module_cache = previous


def paired(adapter, kernels):
    import torch
    directories = [Path.cwd() / name for name in ("incumbent", "challenger")]
    caches = [{}, {}]
    calls = []
    samples = [[], []]
    for i, kernel in enumerate(kernels):
        directories[i].mkdir()
        with chdir(directories[i]), jit_cache(caches[i]):
            fixture, module, device = setup(adapter, kernel, "npu", module_name=f"kernel_{i}")
            call = fixture.benchmark(module, device)
            for _ in range(3):
                call()
            torch.npu.synchronize(device)
            calls.append(call)
    for pair in range(10):
        for i in ((0, 1) if pair % 2 == 0 else (1, 0)):
            with chdir(directories[i]), jit_cache(caches[i]):
                start = time.perf_counter_ns()
                calls[i]()
                torch.npu.synchronize(device)
                samples[i].append((time.perf_counter_ns() - start) / 1e6)
    Path("pairs.json").write_text(json.dumps(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "verify", "profile", "paired", "reference"))
    parser.add_argument("adapter")
    parser.add_argument("kernel", type=Path)
    parser.add_argument("challenger", type=Path, nargs="?")
    args = parser.parse_args()
    if args.phase == "reference":
        paired(args.adapter, (None, args.kernel))
    elif args.phase == "paired":
        if args.challenger is None:
            parser.error("paired requires both an incumbent and a challenger")
        paired(args.adapter, (args.kernel, args.challenger))
    else:
        mode = "sim" if args.phase == "smoke" else "npu"
        fixture, module, device = setup(args.adapter, args.kernel, mode, args.phase == "profile")
        if args.phase == "profile":
            import torch
            fixture.benchmark(module, device)()
            torch.npu.synchronize(device)
        else:
            fixture.check(module, mode, device)
    print(f"PASS: device tune {args.phase}", flush=True)


if __name__ == "__main__":
    main()
