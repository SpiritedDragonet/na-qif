from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ExportConfig


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
    print(f"root: {config.root}")
    print(f"main: {config.main_tex}")
    print(f"template: {config.template_docx}")
    print(f"output: {config.output_docx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

