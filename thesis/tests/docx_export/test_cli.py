import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from docx_export.__main__ import main


def passing_report():
    return SimpleNamespace(ok=True, format=lambda: "ok")


def failing_report():
    return SimpleNamespace(ok=False, format=lambda: "ERROR: bad")


class CliTests(unittest.TestCase):
    def test_runs_pipeline_in_order_when_validation_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "thesis.tex").write_text("% thesis", encoding="utf-8")
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"docx")
            calls = []

            def prepare(config):
                calls.append("prepare")
                return root / "workspace"

            def pandoc(config, workspace):
                calls.append("pandoc")
                return root / "intermediate.docx"

            def merge(template, intermediate, output):
                calls.append("merge")

            def validate(output):
                calls.append("validate")
                return passing_report()

            def refresh(output, enabled):
                calls.append("refresh")
                return SimpleNamespace(attempted=False, succeeded=False, message="disabled")

            with patch("docx_export.__main__.prepare_pandoc_workspace", side_effect=prepare), patch(
                "docx_export.__main__.run_pandoc", side_effect=pandoc
            ), patch("docx_export.__main__.merge_with_template", side_effect=merge), patch(
                "docx_export.__main__.validate_docx", side_effect=validate
            ), patch("docx_export.__main__.refresh_fields", side_effect=refresh):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(["--root", str(root)])

        self.assertEqual(code, 0)
        self.assertEqual(calls, ["prepare", "pandoc", "merge", "validate", "refresh"])

    def test_returns_2_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "thesis.tex").write_text("% thesis", encoding="utf-8")
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"docx")
            with patch("docx_export.__main__.prepare_pandoc_workspace", return_value=root), patch(
                "docx_export.__main__.run_pandoc", return_value=root / "intermediate.docx"
            ), patch("docx_export.__main__.merge_with_template"), patch(
                "docx_export.__main__.validate_docx", return_value=failing_report()
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(["--root", str(root)])

        self.assertEqual(code, 2)

    def test_validate_only_skips_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "existing.docx"
            (root / "thesis.tex").write_text("% thesis", encoding="utf-8")
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"docx")
            with patch("docx_export.__main__.prepare_pandoc_workspace") as prepare, patch(
                "docx_export.__main__.validate_docx", return_value=passing_report()
            ) as validate:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(["--root", str(root), "--output", str(output), "--validate-only"])

        self.assertEqual(code, 0)
        prepare.assert_not_called()
        validate.assert_called_once_with(output.resolve())
