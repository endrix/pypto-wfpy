def english_output(prompt: str) -> str:
    """Add the project's output language to an agent's task instructions."""
    return (
        f"{prompt.rstrip()}\n\n"
        "Follow the skill instructions regardless of their language. Write all replies and "
        "human-readable content in English, including headings, reports, docstrings, and "
        "code comments. Use the required output format and preserve identifiers, API names, "
        "JSON keys, and exact literals required by the workflow."
    )
