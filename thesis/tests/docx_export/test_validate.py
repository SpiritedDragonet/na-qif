import tempfile
import unittest
import zipfile
from pathlib import Path

from docx_export.validate import validate_docx


class ValidateDocxTests(unittest.TestCase):
    def test_detects_latex_residue(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "bad.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr("word/document.xml", "<w:t>\\rm x</w:t>")

            report = validate_docx(docx)

            self.assertFalse(report.ok)
            self.assertIn("LaTeX residue", "\n".join(report.errors))

    def test_detects_unsupported_pdf_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "bad.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr("word/document.xml", "<w:document/>")
                archive.writestr("word/media/figure.pdf", b"%PDF")

            report = validate_docx(docx)

            self.assertFalse(report.ok)
            self.assertIn("Unsupported media", "\n".join(report.errors))

