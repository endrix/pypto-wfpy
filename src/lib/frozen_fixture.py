"""Build an executable candidate against the original standalone script's fixture."""

import ast
from pathlib import Path


def _functions(scope, prefix=()):
    """Index by lexical scope because two factories can define kernels with the same name."""
    for node in scope.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = (*prefix, node.name)
            if isinstance(node, ast.FunctionDef):
                yield name, node
            yield from _functions(node, name)


def checked_source(original: str, proposed: str) -> str:
    """Keep host code, inputs, assertions and tolerances from the initial script."""
    fixture = ast.parse(original)
    edits = ast.parse(proposed)
    functions = dict(_functions(edits))
    kernels = []
    for name, node in _functions(fixture):
        if not any(ast.unparse(d.func if isinstance(d, ast.Call) else d) == "pypto.frontend.jit"
                   for d in node.decorator_list):
            continue
        replacement = functions.get(name)
        if not replacement or ast.dump(replacement.args) != ast.dump(node.args):
            raise ValueError(f"preserve the kernel interface for {'.'.join(name)}")
        node.body = replacement.body
        kernels.append(node)
    if not kernels:
        raise ValueError("the standalone workflow requires a PyPTO JIT kernel to check an edit")

    def assignments(module):
        return {n.targets[0].id: n for n in module.body if isinstance(n, ast.Assign)
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}

    # Tuning a tile must never change a fixture that also reads that constant.
    # Give the JIT its own bindings. On subsequent rounds a checked incumbent
    # already contains those names, so carry its accepted tile values forward.
    original_assignments = assignments(fixture)
    proposed_assignments = assignments(edits)
    aliases = {name: f"_wfpy_tuned_{name}" for name in original_assignments
               if name.isupper() and "TILE" in name}
    for name, alias in aliases.items():
        original_node = original_assignments[name]
        new = proposed_assignments.get(name, original_node)
        if ast.dump(new.value) == ast.dump(original_node.value):
            new = proposed_assignments.get(alias, new)
        value = ast.parse(ast.unparse(new.value), mode="eval").body
        if ast.dump(value) == ast.dump(original_node.value):
            # The original expression may call a shape helper. Evaluate it only once.
            value = ast.Name(id=name, ctx=ast.Load())
        else:
            for element in ast.walk(value):
                if not isinstance(element, (ast.Tuple, ast.List, ast.Load, ast.Name, ast.Constant,
                                            ast.Subscript, ast.BinOp, ast.Mult, ast.FloorDiv, ast.Add, ast.Sub)):
                    raise ValueError(f"unsupported tile expression for {name}")
                if isinstance(element, ast.Name) and element.id in aliases:
                    element.id = aliases[element.id]
        assignment = ast.Assign(targets=[ast.Name(id=alias, ctx=ast.Store())], value=value)
        ast.copy_location(assignment, original_node)
        fixture.body.insert(fixture.body.index(original_node) + 1, assignment)
    for kernel in kernels:
        for statement in kernel.body:
            for element in ast.walk(statement):
                if isinstance(element, ast.Name) and element.id in aliases:
                    element.id = aliases[element.id]
    return ast.unparse(fixture) + "\n"


def bind_fixture(original: str, candidate: str) -> str:
    """Only workflow-owned candidate files are replaced. Keep the proposal for diagnostics."""
    path = Path(candidate)
    proposed = path.read_text()
    path.with_suffix(".proposed.py").write_text(proposed)
    try:
        checked = checked_source(original, proposed)
    except (ValueError, SyntaxError) as exc:
        # Return a failing executable through the normal bounded repair path.
        checked = f"raise ValueError({str(exc)!r})\n"
    path.write_text(checked)
    return str(path)
