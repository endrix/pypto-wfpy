"""Frozen KDA fixture and timing adapter used by the device tuning workflow."""

from tools.check_kda import compare, fixtures, inputs, oracle


def check(module, mode, device):
    for rows, width, diagonal, distribution in fixtures("smoke" if mode == "sim" else "verify"):
        tensors = inputs(rows, width, distribution)
        expected, mask = oracle(*tensors, diagonal)
        actual = module.kda_run(*(x.to(device) for x in tensors), diagonal=diagonal, run_mode=mode)
        if actual.device != device:
            raise ValueError(f"kernel returned {actual.device}, expected {device}")
        compare(actual, expected, mask, f"{mode} ({rows},{width}) diagonal={diagonal} {distribution}")


def benchmark(module, device):
    rows, width, diagonal, distribution = fixtures("score")[0]
    tensors = tuple(x.to(device) for x in inputs(rows, width, distribution))
    return lambda: module.kda_run(*tensors, diagonal=diagonal, run_mode="npu")
