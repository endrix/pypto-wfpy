"""Bisect measured attention checkpoints to localize an error."""

from wfpy import Port, agent

from src.agents import english_output
from src.lib.combo_config import CONNECTOR, MODEL, TIMEOUT_MS, TRANSPORT


@agent(prompt=english_output(
    "Localize the precision loss using the review and measured NPU checkpoint table. "
    "Apply the skill's bisection method to the existing aligned dumps. Cite the tested "
    "midpoints, nearest matching checkpoint and first divergent operation, including "
    "op magic, value difference and the matching source statement. Account for "
    "structural mismatches and possible later cancellation. Return English Markdown."
), skill="precision-binary-search", transport=TRANSPORT, connector=CONNECTOR if TRANSPORT == "acp" else "",
       model=MODEL, timeout_ms=TIMEOUT_MS)
class Diagnose:
    class Ports:
        Evidence = Port[str](direction="in")
        Review = Port[str](direction="in")
        Diagnosis = Port[str](direction="out")
