"""PANKO's dedicated world-model/evaluator agent."""

from src.agents import english_output
from src.lib.combo_config import CONNECTOR, MODEL, TIMEOUT_MS, TRANSPORT
from src.lib.panko import MANUAL
from wfpy import File, Port, agent


@agent(prompt=english_output(
    "You are pypto-op-auto-tuner-panko. Execute one INIT or STEP dispatch from the attached "
    "PANKO skill, using the interpreter, directory, configuration and test command in Request. "
    "Run the vendor harness and frozen NPU evaluator with native tools. Read the on-disk "
    "tree on every dispatch. Leave source edits for the coder except when the harness owns "
    "them. Return a JSON object in Directive: CODE or STOP per the skill. If execution is "
    "blocked, use directive ERROR with the exact diagnostic or dependency question."
), transport=TRANSPORT, connector=CONNECTOR, model=MODEL, timeout_ms=TIMEOUT_MS,
       cli_tools_mode="native", skill=MANUAL)
class PankoTuner:
    class Ports:
        Request = Port[str](direction="in")
        Directive = Port[File](direction="out", ext=".json")
