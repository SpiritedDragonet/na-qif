from __future__ import annotations

import subprocess
from pathlib import Path

from .config import ExportConfig


NUMERIC_CSL = Path(__file__).with_name("gbt7714-numeric.csl")


def build_pandoc_command(config: ExportConfig, workspace: Path, output_docx: Path) -> list[str]:
    command = [
        "pandoc",
        "thesis.tex",
        "--from=latex",
        "--to=docx",
        "--resource-path=.;figures;..;../figures",
        "--number-sections",
        f"--output={output_docx}",
    ]
    if (workspace / "reference.bib").exists():
        command.insert(-1, "--citeproc")
        command.insert(-1, "--bibliography=reference.bib")
        command.insert(-1, f"--csl={NUMERIC_CSL}")
    return command


def run_pandoc(config: ExportConfig, workspace: Path) -> Path:
    output_docx = workspace / "intermediate.docx"
    command = build_pandoc_command(config, workspace, output_docx)
    completed = subprocess.run(command, cwd=workspace, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Pandoc failed with exit code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return output_docx
