"""Evaluate generated residual RMSNorm branches against each other and unfused PyTorch."""

import argparse
from contextlib import chdir
import json
from math import isfinite
import os
from pathlib import Path
from statistics import geometric_mean, median
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib.residual_contract import HELD_OUT_SHAPE, TIMING_SHAPE

WARMUPS = 3
ROUNDS = 12
MIN_GAIN = 0.02
MIN_WINS = 9


def select(samples):
    """Weight the two shapes equally and retain the baseline if the result is noisy."""
    scores = {}
    for name in samples["unfused"]:
        baseline = samples["unfused"][name]
        if len(baseline) != ROUNDS or any(not isfinite(v) or v <= 0 for v in baseline):
            raise ValueError(f"incomplete baseline samples for {name}")
    for method, by_shape in samples.items():
        if method == "unfused":
            continue
        ratios = []
        for shape, baseline in samples["unfused"].items():
            values = by_shape[shape]
            if len(values) != ROUNDS or any(not isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"incomplete {method} samples for {shape}")
            wins = sum(t < b for t, b in zip(values, baseline))
            ratio = median(values) / median(baseline)
            if wins < MIN_WINS or ratio > 1 - MIN_GAIN:
                break
            ratios.append(ratio)
        else:
            scores[method] = geometric_mean(ratios)
    return min(scores, key=scores.get) if scores else "unfused"


def compare(paths):
    import torch
    import torch_npu
    import pypto
    from src.lib.residual_eval import check, check_case, timed_call, unfused
    from tools.check_kda import load_module
    from tools.device_tune_eval import jit_cache

    device = torch.device(f"npu:{os.environ['TILE_FWK_DEVICE_ID']}")
    torch_npu.npu.set_device(device)
    pypto.set_debug_options(runtime_debug_mode=0)
    errors = {}
    caches = {}
    methods = {}
    directories = {}
    for name, path in {"unfused": None, **paths}.items():
        # Each author may use the same function name. Switch its JIT cache
        # before compiling or running, while keeping exclusive device ownership.
        caches[name] = {}
        directories[name] = Path.cwd() / name
        directories[name].mkdir()
        try:
            with chdir(directories[name]), jit_cache(caches[name]):
                module = load_module("residual_" + name, Path(path)) if path else None
                if module is None:
                    def implementation(x, residual, weight, eps, run_mode):
                        return unfused(x, residual, weight, eps=eps)
                else:
                    check(module, "npu", device)
                    implementation = module.residual_rmsnorm
                calls = {}
                for shape in (TIMING_SHAPE, HELD_OUT_SHAPE):
                    check_case(implementation, shape, 43, "npu", device)
                    calls[shape] = timed_call(module, shape, device)
                    for _ in range(WARMUPS):
                        calls[shape]()
                    torch.npu.synchronize(device)
                methods[name] = calls
        except Exception as exc:
            if name == "unfused":
                raise
            errors[name] = f"{type(exc).__name__}: {exc}"

    names = list(methods)
    samples = {name: {} for name in names}
    for shape in (TIMING_SHAPE, HELD_OUT_SHAPE):
        key = "x".join(map(str, shape))
        for name in names:
            samples[name][key] = []
        for i in range(ROUNDS):
            offset = i % len(names)
            for name in names[offset:] + names[:offset]:
                with chdir(directories[name]), jit_cache(caches[name]):
                    start = perf_counter_ns()
                    methods[name][shape]()
                    torch.npu.synchronize(device)
                    samples[name][key].append((perf_counter_ns() - start) / 1e6)
    return {"winner": select(samples), "samples_ms": samples, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path, help="JSON mapping from method name to Python file")
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    paths = json.loads(args.candidates.read_text())
    result = compare(paths)
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: residual RMSNorm comparison")


if __name__ == "__main__":
    main()
