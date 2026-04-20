from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import ExportConfig


TOP_LEVEL_SUFFIXES = {
    ".tex",
    ".bib",
    ".cls",
    ".sty",
    ".cfg",
    ".bst",
    ".ist",
    ".jpg",
    ".jpeg",
    ".png",
    ".wmf",
}
PROJECT_DIRS = ("front", "body", "back", "figures")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _reset_workspace(config: ExportConfig) -> Path:
    work_dir = config.work_dir.resolve()
    allowed_parent = (config.root / ".codex_tmp").resolve()
    if not _is_relative_to(work_dir, allowed_parent):
        raise ValueError(f"Refusing to clean workspace outside .codex_tmp: {work_dir}")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    return work_dir


def normalize_latex_for_pandoc(text: str, root: Path) -> str:
    def replace_figure(match: re.Match[str]) -> str:
        prefix = match.group(1)
        figure_path = match.group(2)
        suffix = match.group(3)
        if not figure_path.lower().endswith(".pdf"):
            return match.group(0)
        png_path = figure_path[:-4] + ".png"
        if (root / png_path).exists():
            return f"{prefix}{png_path}{suffix}"
        return match.group(0)

    text = re.sub(r"(\\+includegraphics(?:\[[^\]]*\])?\{)([^{}]+?)(\})", replace_figure, text)
    replacements = {
        r"\rm": r"\mathrm",
        r"\allowbreak": "",
        r"\Bigl": "",
        r"\Bigr": "",
        r"\bigl": "",
        r"\bigr": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def prepare_pandoc_workspace(config: ExportConfig) -> Path:
    work_dir = _reset_workspace(config)

    for source in config.root.iterdir():
        if source.is_file() and source.suffix.lower() in TOP_LEVEL_SUFFIXES:
            shutil.copy2(source, work_dir / source.name)

    for dirname in PROJECT_DIRS:
        source_dir = config.root / dirname
        if source_dir.exists():
            shutil.copytree(source_dir, work_dir / dirname, dirs_exist_ok=True)

    for tex_file in work_dir.rglob("*.tex"):
        text = tex_file.read_text(encoding="utf-8")
        tex_file.write_text(normalize_latex_for_pandoc(text, work_dir), encoding="utf-8")

    return work_dir

