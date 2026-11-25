"""Propose a scoped statement edit for the measured fault."""

from wfpy import File, Port, agent

from src.agents import english_output
from src.lib.combo_config import CONNECTOR, MODEL, TIMEOUT_MS, TRANSPORT


@agent(prompt=english_output(
    "Repair the diagnosed numerical fault with one executable statement change in "
    "attention_block_kernel. Return one JSON object with string fields old and new. "
    "old must exactly match a supplied source statement, without its line number. "
    "Keep the same signature, output and fixture contract. Use the previous attempt "
    "diagnostics if present."
), skill="pypto-precision-debug", transport=TRANSPORT, connector=CONNECTOR if TRANSPORT == "acp" else "",
       model=MODEL, timeout_ms=TIMEOUT_MS)
class Repair:
    class Ports:
        Evidence = Port[str](direction="in")
        Diagnosis = Port[str](direction="in")
        Edit = Port[File](direction="out", ext=".json")
