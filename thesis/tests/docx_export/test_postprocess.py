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

    def test_copies_external_hyperlink_relationships_used_by_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:sectPr/></w:body></w:document>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:p><w:hyperlink r:id="rId99"><w:r><w:t>link</w:t></w:r></w:hyperlink></w:p></w:body></w:document>"""
            write_docx(
                template,
                {
                    "[Content_Types].xml": content_types,
                    "word/document.xml": template_document,
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                },
            )
            write_docx(
                intermediate,
                {
                    "[Content_Types].xml": content_types,
                    "word/document.xml": intermediate_document,
                    "word/_rels/document.xml.rels": (
                        f"<Relationships xmlns='{REL}'>"
                        "<Relationship Id='rId99' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink' Target='https://example.com' TargetMode='External'/>"
                        "</Relationships>"
                    ),
                },
            )

            merge_with_template(template, intermediate, output)

            with zipfile.ZipFile(output) as archive:
                document = archive.read("word/document.xml").decode("utf-8")
                rels = archive.read("word/_rels/document.xml.rels").decode("utf-8")
                self.assertIn("https://example.com", rels)
                self.assertNotIn('r:id="rId99"', document)

    def test_copies_numbering_definitions_used_by_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:sectPr/></w:body></w:document>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:p><w:pPr><w:numPr><w:numId w:val="1001"/></w:numPr></w:pPr><w:r><w:t>item</w:t></w:r></w:p></w:body></w:document>"""
            template_numbering = f"""<w:numbering xmlns:w="{W}"/>"""
            intermediate_numbering = f"""<w:numbering xmlns:w="{W}"><w:abstractNum w:abstractNumId="99201"><w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum><w:num w:numId="1001"><w:abstractNumId w:val="99201"/></w:num></w:numbering>"""
            write_docx(
                template,
                {
                    "[Content_Types].xml": content_types,
                    "word/document.xml": template_document,
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                    "word/numbering.xml": template_numbering,
                },
            )
            write_docx(
                intermediate,
                {
                    "[Content_Types].xml": content_types,
                    "word/document.xml": intermediate_document,
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                    "word/numbering.xml": intermediate_numbering,
                },
            )

            merge_with_template(template, intermediate, output)

            with zipfile.ZipFile(output) as archive:
                numbering = archive.read("word/numbering.xml").decode("utf-8")
                self.assertIn('w:numId="1001"', numbering)
                self.assertIn('w:abstractNumId="99201"', numbering)
