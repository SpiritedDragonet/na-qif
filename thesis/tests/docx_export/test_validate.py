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

    def test_detects_unresolved_equation_reference_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "bad.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr("word/document.xml", "<w:t>由式 [eq:sample] 可得。</w:t>")

            report = validate_docx(docx)

            self.assertFalse(report.ok)
            self.assertIn("Unresolved equation reference found: [eq:sample]", report.errors)

    def test_detects_missing_document_relationship(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "bad.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main' "
                    "xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'>"
                    "<w:body><w:p><w:hyperlink r:id='rId99'/></w:p></w:body></w:document>",
                )
                archive.writestr(
                    "word/_rels/document.xml.rels",
                    "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'/>",
                )

            report = validate_docx(docx)

            self.assertFalse(report.ok)
            self.assertIn("Missing document relationship for rId99", report.errors)

    def test_detects_missing_numbering_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "bad.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
                    "<w:body><w:p><w:pPr><w:numPr><w:numId w:val='1001'/></w:numPr></w:pPr></w:p></w:body></w:document>",
                )
                archive.writestr(
                    "word/numbering.xml",
                    "<w:numbering xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'/>",
                )

            report = validate_docx(docx)

            self.assertFalse(report.ok)
            self.assertIn("Missing numbering definition for numId 1001", report.errors)

    def test_detects_undeclared_ignorable_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "bad.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main' "
                    "xmlns:mc='http://schemas.openxmlformats.org/markup-compatibility/2006' "
                    "mc:Ignorable='w14'><w:body/></w:document>",
                )

            report = validate_docx(docx)

            self.assertFalse(report.ok)
            self.assertIn("Undeclared mc:Ignorable prefix w14 in word/document.xml", report.errors)

    def test_detects_unlinked_author_year_citation_when_bibliography_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "bad.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
                    "<w:body>"
                    "<w:p><w:r><w:t>See (Nielsen and Chuang 2010).</w:t></w:r></w:p>"
                    "<w:p><w:pPr><w:pStyle w:val='a'/></w:pPr><w:r><w:t>Nielsen, Michael A., and Isaac L. Chuang. 2010. Book.</w:t></w:r></w:p>"
                    "</w:body></w:document>",
                )

            report = validate_docx(docx)

            self.assertFalse(report.ok)
            self.assertIn("Citation is not linked to bibliography: Nielsen and Chuang 2010", report.errors)
