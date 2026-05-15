from __future__ import annotations

import argparse
import sys

from .config import ExportConfig
from .pandoc_runner import run_pandoc
from .postprocess import merge_with_template, normalize_docx_styles
from .preprocess import prepare_pandoc_workspace
from .validate import validate_docx
from .word_refresh import refresh_fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the thesis LaTeX project to DOCX.")
    parser.add_argument("--root", default=".", help="Thesis project root.")
    parser.add_argument("--output", help="Output DOCX path.")
    parser.add_argument("--template", help="Template DOCX path.")
    parser.add_argument("--refresh-fields", action="store_true", help="Refresh Word fields through COM when available.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the output DOCX without regenerating it.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ExportConfig.from_root(
        args.root,
        output=args.output,
        template=args.template,
        refresh_fields=args.refresh_fields,
    )
    if args.validate_only:
        report = validate_docx(config.output_docx)
        print(report.format())
        return 0 if report.ok else 2

    workspace = prepare_pandoc_workspace(config)
    intermediate_docx = run_pandoc(config, workspace)
    merge_with_template(config.template_docx, intermediate_docx, config.output_docx, source_root=config.root)
    report = validate_docx(config.output_docx)
    print(report.format())
    if not report.ok:
        return 2

    refresh_status = refresh_fields(config.output_docx, config.refresh_fields)
    if refresh_status.attempted:
        print(refresh_status.message)
    if refresh_status.succeeded:
        normalize_docx_styles(config.output_docx)
        report = validate_docx(config.output_docx)
        print(report.format())
        if not report.ok:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
