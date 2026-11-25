"""Materialize a scalar workflow output for the IDE's file viewer."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from wfpy import File, Port, action, task


@task
class WriteViewerText:
    directory: str
    label: str

    class Ports:
        Text = Port[str](direction="in")
        Document = Port[File](direction="out", ext=".md")

    @action(consumes={"Text": 1}, produces={"Document": 1})
    def write(self, text: str) -> str:
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                prefix=f"{self.label}-", suffix=".md", delete=False) as doc:
            doc.write(text.rstrip("\n") + "\n")
        return doc.name
