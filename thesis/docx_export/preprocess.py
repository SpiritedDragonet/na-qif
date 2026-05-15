from __future__ import annotations

import re
import shutil
import subprocess
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
BIBLIOGRAPHY_MARKER = "DOCX_EXPORT_BIBLIOGRAPHY_MARKER"


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


def _graphic_candidates(root: Path, figure_path: str) -> list[Path]:
    path = Path(figure_path)
    if path.is_absolute():
        return [path]
    return [root / path, root / "figures" / path]


def _first_existing_graphic(root: Path, figure_path: str) -> Path | None:
    for candidate in _graphic_candidates(root, figure_path):
        if candidate.exists():
            return candidate
    return None


def _render_pdf_figure_to_png(pdf_path: Path) -> None:
    png_stem = pdf_path.with_suffix("")
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", str(pdf_path), str(png_stem)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("DOCX export needs pdftoppm to render PDF figures to PNG.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to render PDF figure for DOCX export: "
            f"{pdf_path}\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}"
        ) from exc


def _ensure_png_for_pdf_figure(root: Path, figure_path: str) -> str | None:
    png_path = figure_path[:-4] + ".png"
    if _first_existing_graphic(root, png_path):
        return png_path

    pdf_path = _first_existing_graphic(root, figure_path)
    if pdf_path is None:
        return None

    _render_pdf_figure_to_png(pdf_path)
    return png_path


def normalize_latex_for_pandoc(text: str, root: Path) -> str:
    def replace_figure(match: re.Match[str]) -> str:
        prefix = match.group(1)
        figure_path = match.group(2)
        suffix = match.group(3)
        if not figure_path.lower().endswith(".pdf"):
            return match.group(0)
        png_path = _ensure_png_for_pdf_figure(root, figure_path)
        if png_path is not None:
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
    text = _remove_heu_trailing_heading_titles(text)
    text = _replace_bibliography_commands(text)
    return text


def _remove_heu_trailing_heading_titles(text: str) -> str:
    heading = r"\\(?:chapter|section|subsection|subsubsection)\*?(?:\[[^\]\n]*\])?\{(?:[^{}]|\{[^{}]*\})*\}"
    return re.sub(rf"({heading})\[[^\]\n]*\]", r"\1", text)


def _replace_bibliography_commands(text: str) -> str:
    text = re.sub(r"(?m)^[ \t]*\\bibliographystyle\{[^{}\n]+\}[ \t]*\n?", "", text)
    return re.sub(r"\\bibliography\{[^{}\n]+\}", f"\\\\section*{{参考文献}}\n\n{BIBLIOGRAPHY_MARKER}", text)


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
