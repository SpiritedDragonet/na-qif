import tempfile
import unittest
import zipfile
from pathlib import Path

from docx_export.postprocess import merge_with_template


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def write_docx(path: Path, parts: dict[str, str | bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in parts.items():
            archive.writestr(name, content)


class PostprocessTests(unittest.TestCase):
    def test_merges_pandoc_body_into_template_and_preserves_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            styles = b"<w:styles xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'/>"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:sectPr/></w:body></w:document>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:p><w:r><w:t>hello</w:t></w:r></w:p></w:body></w:document>"""
            write_docx(
                template,
                {
                    "[Content_Types].xml": content_types,
                    "word/document.xml": template_document,
                    "word/styles.xml": styles,
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                },
            )
            write_docx(
                intermediate,
                {
                    "[Content_Types].xml": content_types,
                    "word/document.xml": intermediate_document,
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                },
            )

            merge_with_template(template, intermediate, output)

            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read("word/styles.xml"), styles)
                document = archive.read("word/document.xml").decode("utf-8")
                self.assertIn("hello", document)
                self.assertIn("TOC", document)

