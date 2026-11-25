"""Run one KDA evaluation phase with fixtures owned by the workflow."""

import argparse
import importlib.util
import os
from pathlib import Path
from statistics import median
import sys
import time

import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Retries can replace a source with the same size and timestamp.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def fixtures(phase):
    smoke = [(17, 7, 1, "cumulative"), (2, 1, 0, "wide-step")]
    score = [(128, 64, 1, "cumulative")]
    if phase in ("golden", "verify", "final"):
        return score + smoke + [(17, 65, 0, "cumulative"), (2, 1, 1, "wide-step")]
    return smoke if phase == "smoke" else score


def inputs(bt, width, distribution):
    rng = torch.Generator().manual_seed(0)
    a, k = (torch.randn(bt, width, generator=rng) for _ in range(2))
    g = (-1.5 * torch.rand(bt, width, generator=rng)).cumsum(0)
    if distribution == "wide-step":
        g.zero_()
        g[1:] = -1000
    return a, k, g


def oracle(a, k, g, diagonal):
    # Mask before exponentiating: discarded upper entries can overflow.
    mask = torch.ones(len(a), len(a), dtype=torch.bool).triu(diagonal)
    g = g.double()
    decay = (g[:, None] - g[None, :]).masked_fill(mask[..., None], 0).exp()
    out = (a.double()[:, None] * decay * k.double()[None, :]).sum(-1)
    return out.masked_fill(mask, 0), mask


def compare(actual, expected, mask, label):
    actual = actual.detach().cpu()
    if actual.dtype != torch.float32 or actual.shape != expected.shape:
        raise ValueError(f"{label}: got {actual.dtype} {tuple(actual.shape)}, expected FP32 {tuple(expected.shape)}")
    invalid = ~torch.isfinite(actual) | (mask & (actual != 0))
    if invalid.any():
        row, col = invalid.nonzero()[0].tolist()
        raise ValueError(f"{label}: [{row},{col}] got {actual[row, col].item()}, "
                         f"expected {expected[row, col].item()} (masked={mask[row, col].item()})")
    error = (actual.double() - expected.double()).abs().max().item()
    print(f"{label} max_abs_error={error:g}", flush=True)
    torch.testing.assert_close(actual.double(), expected.double(), atol=1e-3, rtol=1e-3)


def check_cases(phase, device, mode, golden, run, saved):
    """Check deterministic fixtures and retain each golden tensor for kernel verification."""
    for fixture in fixtures(phase):
        bt, width, diagonal, distribution = fixture
        start = time.monotonic()
        case = f"{phase} shape=({bt},{width}) diagonal={diagonal} seed=0 gates={distribution}"
        print(f"START {case}", flush=True)
        tensors = inputs(bt, width, distribution)
        expected, mask = oracle(*tensors, diagonal)
        expected_golden = None
        if golden is not None:
            expected_golden = golden(*(x.clone().to(device) for x in tensors), diagonal=diagonal).detach().cpu()
            compare(expected_golden, expected, mask, "golden vs FP64")
            if phase == "golden":
                saved[fixture] = expected_golden
        elif phase in ("verify", "final"):
            expected_golden = saved[fixture]
        if run is not None:
            actual = run(*(x.to(device) for x in tensors), diagonal=diagonal, run_mode=mode)
            if actual.device != device:
                raise ValueError(f"kernel returned {actual.device}, expected {device}")
            compare(actual, expected, mask, "kernel vs FP64")
            if expected_golden is not None:
                compare(actual, expected_golden, mask, "kernel vs golden")
        print(f"COMPLETE {case} elapsed={time.monotonic()-start:.2f}s", flush=True)


def time_npu(run, device):
    bt, width, diagonal, distribution = fixtures("score")[0]
    tensors = tuple(x.to(device) for x in inputs(bt, width, distribution))
    for _ in range(3):
        run(*tensors, diagonal=diagonal, run_mode="npu")
    torch.npu.synchronize(device)
    samples = []
    for _ in range(10):
        start = time.perf_counter_ns()
        run(*tensors, diagonal=diagonal, run_mode="npu")
        torch.npu.synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    print(f"NPU latency: median {median(samples):.3f} ms per synchronized kda_run "
          f"({bt}x{width}, seed 0, diagonal {diagonal}, 3 warmups, 10 samples)", flush=True)
    print(f"NPU latency samples (ms): {samples}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "score", "golden", "verify", "final"))
    parser.add_argument("kernel", type=Path)
    parser.add_argument("golden", type=Path)
    parser.add_argument("--golden-outputs", type=Path,
                        help="Save validated golden tensors here, or load them for kernel verification")
    args = parser.parse_args()
    mode = "npu" if args.phase in ("golden", "verify", "final") else "sim"
    device = torch.device("cpu")
    if mode == "npu":
        import torch_npu
        device = torch.device(f"npu:{os.environ['TILE_FWK_DEVICE_ID']}")
        torch_npu.npu.set_device(device)
    else:
        import pypto
        pypto.set_debug_options(dump_pass_graph=["InferTensorFormat", "ExpandFunction", "SubgraphToFunction", "CodegenPreproc"])

    sys.path.insert(0, str(args.golden.parent))
    run = None if args.phase == "golden" else load_module("authored_kernel", args.kernel).kda_run
    saved = {}
    golden = None
    if args.phase in ("verify", "final") and args.golden_outputs:
        saved = torch.load(args.golden_outputs, map_location="cpu", weights_only=True)
    elif mode == "npu":
        golden = load_module("authored_golden", args.golden).kda_golden
    check_cases(args.phase, device, mode, golden, run, saved)
    if args.phase == "final":
        time_npu(run, device)
    if args.phase == "golden" and args.golden_outputs:
        torch.save(saved, args.golden_outputs)
    print(f"PASS: KDA {args.phase}", flush=True)


if __name__ == "__main__":
    main()
