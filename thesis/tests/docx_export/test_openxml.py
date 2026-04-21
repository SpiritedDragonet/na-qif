import tempfile
import unittest
import zipfile
from pathlib import Path

from xml.etree import ElementTree as ET

from docx_export.openxml import DocxPackage, MC_NS, W15_NS, W_NS, qn


class OpenXmlPackageTests(unittest.TestCase):
    def test_reads_and_writes_docx_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.docx"
            target = Path(tmp) / "target.docx"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types/>")
                archive.writestr("word/document.xml", "<w:document/>")

            package = DocxPackage.read(source)
            package.parts["word/document.xml"] = b"<w:document><w:body/></w:document>"
            package.write(target)

            with zipfile.ZipFile(target) as archive:
                self.assertEqual(archive.read("word/document.xml"), b"<w:document><w:body/></w:document>")

    def test_next_relationship_id_skips_existing_ids(self):
        package = DocxPackage(parts={})
        rels_xml = b'''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1"/><Relationship Id="rId3"/></Relationships>'''

        self.assertEqual(package.next_relationship_id(rels_xml), "rId4")

    def test_set_xml_part_repairs_undeclared_ignorable_prefixes(self):
        root = ET.Element(qn(W_NS, "numbering"), {qn(MC_NS, "Ignorable"): "w14 w15"})
        ET.SubElement(root, qn(W_NS, "abstractNum"), {qn(W_NS, "abstractNumId"): "1", qn(W15_NS, "restartNumberingAfterBreak"): "0"})
        package = DocxPackage(parts={})

        package.set_xml_part("word/numbering.xml", root)

        xml = package.parts["word/numbering.xml"].decode("utf-8")
        self.assertIn("xmlns:w15=", xml)
        self.assertIn('mc:Ignorable="w15"', xml)
        self.assertNotIn("w14 w15", xml)
