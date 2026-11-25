"""Apply a single PANKO intent to the selected parent source."""

from src.agents import english_output
from src.lib.combo_config import CONNECTOR, MODEL, TIMEOUT_MS, TRANSPORT
from wfpy import File, Port, agent


@agent(prompt=english_output(
    "Apply only the supplied intent to the supplied parent kernel. Preserve its "
    "host wrapper, public interface, fixtures and instrumentation. Return the complete "
    "Python module in Source, without Markdown fences."
), transport=TRANSPORT, connector=CONNECTOR, model=MODEL, timeout_ms=TIMEOUT_MS)
class PankoCoder:
    class Ports:
        Request = Port[str](direction="in")
        Source = Port[File](direction="out", ext=".py")
