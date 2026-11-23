#!/usr/bin/env python3
# coding: utf-8
"""Instrument a kernel for pass verification, and read what it dumps.

PyPTO can dump the output tensor of **every op** it lowers. Turning that on is
one edit to the `@pypto.frontend.jit` decorator:

    verify_options = {"enable_pass_verify": True,
                      "pass_verify_save_tensor": True,
                      "pass_verify_pass_filter": []}

    @pypto.frontend.jit(runtime_options=runtime_options, verify_options=verify_options)

The run then writes, under `<work>/output/output_*/verify/`:

- `verify_graph_data_metainfo.csv` -- one row per dumped tensor, with its
  opcode, shape, dtype and file name
- `tensor_graph/<...>~<opmagic>~<OPCODE>~...data` -- the raw bytes

`pypto-precision-compare` also describes setting a golden with
`set_verify_golden_data`, which makes PyPTO judge the final output itself. That
is not needed here and is deliberately not done: a golden has to be set between
computing the reference and calling the kernel, which means rewriting the
kernel's `main()`, and that is specific to each file. Measured: the per-op dumps
appear without any golden -- 26 of them on the attention kernel -- because the
golden only drives the final PASS/FAIL verdict, not the dumping.

Two constraints found by measurement:

- **NPU only.** In sim the run dies in `setup_verify_data` with
  `RT_MEMCPY_FAILED`: copying the golden wants a real device.
- **A wrong reduction axis does not change the op sequence.** Good and broken
  builds of the same kernel dumped 20 identically-named ops. So the fault shows
  in the *values*, not the structure, and localising it means comparing tensors
  op by op.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.lib.jit_options import add_option

from src.lib.records import OpDiff, OpDump

# numpy is imported where it is used, not here. `wfpy plan` -- which the IDE runs
# constantly -- imports a workflow module IN-PROCESS, with whatever interpreter
# is running wfpy. That is often not the project's venv: the IDE was seen
# planning with `workflow/wfpy/.venv/bin/wfpy`, which has wfpy but no numpy, and
# a module-level import there turns "show me the graph" into
# ModuleNotFoundError. Nothing about drawing the graph needs numpy; only
# comparing tensors does.

# The metainfo's dtype names, as numpy type names. Strings rather than numpy
# objects so this module still imports without numpy; resolved in _values().
# A dtype not here is readable as bytes but not comparable, so its row is
# reported as unread rather than as equal.
DTYPES = {
    "FP16": "float16",
    "FP32": "float32",
    "INT32": "int32",
    "INT8": "int8",
    "UINT8": "uint8",
    "INT64": "int64",
}

# Below this, two runs of the same op agree. The dumps are raw device output, so
# an op can differ in the last bit without the kernel being wrong.
AGREE_ATOL = 1e-3


def instrument(source: str) -> str:
    """Add `verify_options` to a kernel's jit decorator.

    One edit, and generic: it does not touch `main()` or the kernel body. See
    `src.lib.jit_options` for why it is a decorator parameter and why this writes
    a copy.
    """
    return add_option(
        source,
        "verify_options",
        '{"enable_pass_verify": True, "pass_verify_save_tensor": True,\n'
        '                  "pass_verify_pass_filter": []}',
        note="Pass verification: dump every op's output tensor.",
    )


def find_verify_dir(work_dir: str | Path) -> str:
    """The newest `verify` directory under a run's work dir, or ""."""
    candidates = sorted(
        Path(work_dir).glob("output/*/verify"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(candidates[-1]) if candidates else ""


def read_dumps(verify_dir: str | Path) -> dict[str, OpDump]:
    """Every op the tensor_graph phase dumped, keyed by its op magic.

    Keyed on `:opmagic` rather than on the file name: the name carries a
    timestamp, so it differs between two runs of the same kernel while the op
    magic does not. That is what makes two runs alignable at all.
    """
    root = Path(verify_dir)
    meta = root / "verify_graph_data_metainfo.csv"
    if not meta.is_file():
        return {}

    out: dict[str, OpDump] = {}
    for row in csv.DictReader(meta.read_text(encoding="utf-8").splitlines()):
        if (row.get("PHASE_NAME") or "").strip() != "tensor_graph":
            continue
        name = (row.get("FILENAME") or "").strip()
        magic = (row.get(":opmagic") or "").strip()
        if not name.endswith(".data") or not magic:
            continue
        out[magic] = OpDump(
            magic=magic,
            opcode=(row.get(":opcode") or "").strip(),
            shape=(row.get(":shape") or "").strip(),
            dtype=(row.get(":datatype") or "").strip(),
            path=str(root / "tensor_graph" / name),
        )
    return out


def _values(dump: OpDump) -> Any:
    """The dump's bytes as a float32 array, or None if it cannot be read."""
    import numpy as np  # local: see the note on DTYPES above

    name = DTYPES.get(dump.dtype)
    path = Path(dump.path)
    if name is None or not path.is_file():
        return None
    raw = np.frombuffer(path.read_bytes(), dtype=getattr(np, name))
    # float32 throughout: subtracting two fp16 arrays overflows to inf where the
    # values are large, which would read as a divergence.
    return raw.astype(np.float32)


def _order(magic: str) -> int:
    """Ops execute in op-magic order; sort numerically, not as text."""
    return int(magic) if magic.isdigit() else 0


def compare(reference: dict[str, OpDump], suspect: dict[str, OpDump]) -> list[OpDiff]:
    """Every op the two runs share, in execution order, with its difference.

    Only ops present in both are comparable. An op in one run and not the other
    is reported with `comparable=False` rather than dropped, because a structural
    difference between the two builds is itself worth seeing.
    """
    diffs: list[OpDiff] = []
    for magic in sorted(set(reference) | set(suspect), key=_order):
        ref, sus = reference.get(magic), suspect.get(magic)
        if ref is None or sus is None:
            present = "reference only" if sus is None else "suspect only"
            dump = ref or sus
            assert dump is not None
            diffs.append(OpDiff(magic=magic, opcode=dump.opcode, shape=dump.shape,
                                max_abs=0.0, comparable=False, note=present))
            continue

        import numpy as np  # local: see the note on DTYPES above

        a, b = _values(ref), _values(sus)
        if a is None or b is None:
            diffs.append(OpDiff(magic=magic, opcode=ref.opcode, shape=ref.shape,
                                max_abs=0.0, comparable=False,
                                note=f"unreadable ({ref.dtype})"))
            continue
        if a.shape != b.shape:
            diffs.append(OpDiff(magic=magic, opcode=ref.opcode, shape=ref.shape,
                                max_abs=0.0, comparable=False,
                                note=f"different sizes {a.size} vs {b.size}"))
            continue

        diffs.append(OpDiff(magic=magic, opcode=ref.opcode, shape=ref.shape,
                            max_abs=float(np.abs(a - b).max()), comparable=True))
    return diffs


def first_divergence(diffs: list[OpDiff], atol: float = AGREE_ATOL) -> OpDiff | None:
    """The earliest op whose output differs. That is the one to look at.

    Earliest, not largest: the difference grows as it propagates, so the biggest
    number is usually the final output and says nothing about the cause.
    """
    return next((d for d in diffs if d.comparable and d.max_abs > atol), None)
