"""Trusted device fixture driver for the attention precision-repair workflow."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys

GOLDEN = Path(__file__).resolve().parents[1] / "examples/05_attention_block.py"
SEEDS = (0, 17, 29)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    import torch
    import torch_npu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
    torch_npu.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")

    golden = load("precision_reference", GOLDEN)
    candidate = load("precision_candidate", args.candidate)
    shapes = [(golden.SEQ, golden.DIM), (golden.DIM, golden.SEQ),
              (golden.SEQ, golden.DIM), (golden.DIM, golden.FF), (golden.FF, golden.DIM)]
    for seed in (SEEDS if args.seed is None else (args.seed,)):
        torch.manual_seed(seed)
        tensors = [torch.randn(shape, dtype=torch.float16, device=device) for shape in shapes]
        # The original failing fixture uses unit-scale inputs. The extra seeds
        # exercise distinct attention distributions with lower feed-forward gain.
        if seed:
            tensors[-2:] = [w * 0.25 for w in tensors[-2:]]
        expected = golden.reference(*tensors)
        out = torch.full_like(tensors[0], float("nan"))
        candidate.attention_block_kernel(*tensors, out)
        torch_npu.npu.synchronize()
        try:
            torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
        except AssertionError as exc:
            print(f"FAIL: precision fixture seed={seed}\n{exc}", flush=True)
            return 2
        print(f"PASS: precision fixture seed={seed} NPU {device_id}", flush=True)
    print("PASS: all requested precision fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
