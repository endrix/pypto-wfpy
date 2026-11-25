"""Map JIT statements to measured NPU checkpoints."""

from wfpy import Port, agent

from src.agents import english_output
from src.lib.combo_config import CONNECTOR, MODEL, TIMEOUT_MS, TRANSPORT


@agent(prompt=english_output(
    "Inspect the supplied numbered JIT statements and measured checkpoint table. "
    "Return a short ordered mapping from PyPTO call sites to candidate checkpoints, "
    "including source lines, shapes and reduction axes. Return English Markdown."
), skill="pypto-op-review", transport=TRANSPORT, connector=CONNECTOR if TRANSPORT == "acp" else "",
       model=MODEL, timeout_ms=TIMEOUT_MS)
class ReviewCalls:
    class Ports:
        Evidence = Port[str](direction="in")
        Review = Port[str](direction="out")
