from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportConfig:
    root: Path
    main_tex: Path
    template_docx: Path
    output_docx: Path
    work_dir: Path
    refresh_fields: bool = False

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        output: str | Path | None = None,
        template: str | Path | None = None,
        refresh_fields: bool = False,
    ) -> "ExportConfig":
        root_path = Path(root).resolve()
        main_tex = root_path / "thesis.tex"
        template_docx = Path(template).resolve() if template else root_path / "硕士毕业论文参考模板.docx"
        output_docx = Path(output).resolve() if output else root_path / "thesis.docx"
        if not main_tex.exists():
            raise FileNotFoundError(f"Missing thesis.tex: {main_tex}")
        if not template_docx.exists():
            raise FileNotFoundError(f"Missing template DOCX: {template_docx}")
        return cls(
            root=root_path,
            main_tex=main_tex,
            template_docx=template_docx,
            output_docx=output_docx,
            work_dir=root_path / ".codex_tmp" / "docx_export",
            refresh_fields=refresh_fields,
        )
