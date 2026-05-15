import tempfile
import unittest
import zipfile
from pathlib import Path

from docx_export.postprocess import merge_with_template, normalize_docx_styles


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
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

    def test_normalizes_styles_after_word_refresh_rewrites_theme_fonts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docx = root / "word_refreshed.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:pPr><w:pStyle w:val="aa"/><w:spacing w:before="156" w:after="156"/></w:pPr><w:r><w:t>图1.1 系统图</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="a2"/><w:spacing w:before="120" w:after="120" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:t>表1.1 参数表</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Normal"/><w:spacing w:line="360" w:lineRule="auto"/></w:pPr><w:r><w:t>正文。</w:t></w:r></w:p>
<w:sectPr/></w:body></w:document>"""
            styles = f"""<w:styles xmlns:w="{W}">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:spacing w:line="220" w:lineRule="auto"/></w:pPr><w:rPr><w:szCs w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="aa"><w:name w:val="毕设图题"/><w:pPr/><w:rPr><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="a2"><w:name w:val="图表"/><w:pPr/><w:rPr><w:szCs w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="af"><w:name w:val="公式新标准"/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:rFonts w:asciiTheme="minorEastAsia" w:eastAsiaTheme="minorEastAsia" w:hAnsiTheme="minorEastAsia"/></w:rPr></w:style>
</w:styles>"""
            write_docx(
                docx,
                {
                    "[Content_Types].xml": content_types,
                    "word/document.xml": document,
                    "word/styles.xml": styles,
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                },
            )

            normalize_docx_styles(docx)

            with zipfile.ZipFile(docx) as archive:
                styles_xml = archive.read("word/styles.xml")
                document_xml = archive.read("word/document.xml")
            import xml.etree.ElementTree as ET

            root_xml = ET.fromstring(styles_xml)
            document_root = ET.fromstring(document_xml)
            styles_by_name = {
                style.find("w:name", {"w": W}).attrib[f"{{{W}}}val"]: style
                for style in root_xml.findall("w:style", {"w": W})
            }
            paragraphs = document_root.findall(".//w:p", {"w": W})
            figure_caption = paragraphs[0]
            table_caption = paragraphs[1]
            body = paragraphs[2]
            normal = styles_by_name["Normal"]
            figure = styles_by_name["毕设图题"]
            table = styles_by_name["图表"]
            formula = styles_by_name["公式新标准"]
            normal_fonts = normal.find("w:rPr/w:rFonts", {"w": W})
            formula_fonts = formula.find("w:rPr/w:rFonts", {"w": W})
            formula_spacing = formula.find("w:pPr/w:spacing", {"w": W})
            formula_indent = formula.find("w:pPr/w:ind", {"w": W})
            figure_jc = figure.find("w:pPr/w:jc", {"w": W})
            figure_indent = figure.find("w:pPr/w:ind", {"w": W})
            table_indent = table.find("w:pPr/w:ind", {"w": W})
            figure_size = figure.find("w:rPr/w:sz", {"w": W})
            table_size = table.find("w:rPr/w:sz", {"w": W})
            self.assertEqual(normal.find("w:pPr/w:spacing", {"w": W}).attrib[f"{{{W}}}line"], "440")
            self.assertEqual(normal_fonts.attrib[f"{{{W}}}eastAsia"], "宋体")
            self.assertEqual(normal_fonts.attrib[f"{{{W}}}ascii"], "Times New Roman")
            self.assertEqual(figure_jc.attrib[f"{{{W}}}val"], "center")
            self.assertEqual(figure_indent.attrib[f"{{{W}}}firstLine"], "0")
            self.assertEqual(table_indent.attrib[f"{{{W}}}firstLine"], "0")
            self.assertEqual(figure_size.attrib[f"{{{W}}}val"], "21")
            self.assertEqual(table_size.attrib[f"{{{W}}}val"], "21")
            self.assertEqual(formula_spacing.attrib[f"{{{W}}}line"], "240")
            self.assertEqual(formula_spacing.attrib[f"{{{W}}}lineRule"], "auto")
            self.assertEqual(formula_indent.attrib[f"{{{W}}}firstLine"], "0")
            self.assertEqual(formula_fonts.attrib[f"{{{W}}}eastAsia"], "宋体")
            self.assertEqual(formula_fonts.attrib[f"{{{W}}}ascii"], "Times New Roman")
            self.assertNotIn(f"{{{W}}}asciiTheme", formula_fonts.attrib)
            self.assertIsNone(figure_caption.find("w:pPr/w:spacing", {"w": W}))
            self.assertIsNone(table_caption.find("w:pPr/w:spacing", {"w": W}))
            self.assertIsNotNone(body.find("w:pPr/w:spacing", {"w": W}))

    def test_formats_second_and_third_level_headings_without_bold_or_indent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:sectPr/></w:body></w:document>"""
            styles = f"""<w:styles xmlns:w="{W}">
<w:style w:type="paragraph" w:styleId="2"><w:name w:val="heading 2"/><w:pPr><w:ind w:left="720" w:firstLine="480" w:hanging="240"/></w:pPr><w:rPr><w:rFonts w:ascii="Arial" w:eastAsia="宋体" w:hAnsi="Arial"/><w:b/><w:bCs/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="3"><w:name w:val="heading 3"/><w:pPr><w:ind w:left="720" w:firstLine="480"/></w:pPr><w:rPr><w:rFonts w:ascii="Arial" w:eastAsia="宋体" w:hAnsi="Arial"/><w:b/><w:bCs/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style>
</w:styles>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>1.1量子计算简介</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading3"/></w:pPr><w:r><w:t>1.1.1量子计算基本概念与模型</w:t></w:r></w:p>
</w:body></w:document>"""
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

            import xml.etree.ElementTree as ET

            with zipfile.ZipFile(output) as archive:
                output_styles = ET.fromstring(archive.read("word/styles.xml"))
                output_document = ET.fromstring(archive.read("word/document.xml"))

            for style_id in ("2", "3"):
                style = next(item for item in output_styles.findall("w:style", {"w": W}) if item.attrib.get(f"{{{W}}}styleId") == style_id)
                fonts = style.find("w:rPr/w:rFonts", {"w": W})
                self.assertEqual(fonts.attrib.get(f"{{{W}}}ascii"), "Times New Roman")
                self.assertEqual(fonts.attrib.get(f"{{{W}}}hAnsi"), "Times New Roman")
                self.assertEqual(fonts.attrib.get(f"{{{W}}}eastAsia"), "黑体")
                self.assertEqual(style.find("w:rPr/w:b", {"w": W}).attrib.get(f"{{{W}}}val"), "0")
                self.assertEqual(style.find("w:rPr/w:bCs", {"w": W}).attrib.get(f"{{{W}}}val"), "0")
                indent = style.find("w:pPr/w:ind", {"w": W})
                self.assertEqual(indent.attrib.get(f"{{{W}}}firstLine"), "0")
                self.assertNotIn(f"{{{W}}}left", indent.attrib)
                self.assertNotIn(f"{{{W}}}hanging", indent.attrib)

            paragraphs = {
                "".join(text.text or "" for text in paragraph.findall(".//w:t", {"w": W})): paragraph
                for paragraph in output_document.findall(".//w:body/w:p", {"w": W})
            }
            for text, style_id in {"1.1 量子计算简介": "2", "1.1.1 量子计算基本概念与模型": "3"}.items():
                paragraph = paragraphs[text]
                self.assertEqual(paragraph.find("w:pPr/w:pStyle", {"w": W}).attrib.get(f"{{{W}}}val"), style_id)
                indent = paragraph.find("w:pPr/w:ind", {"w": W})
                self.assertEqual(indent.attrib.get(f"{{{W}}}firstLine"), "0")
                run = paragraph.find("w:r", {"w": W})
                fonts = run.find("w:rPr/w:rFonts", {"w": W})
                self.assertEqual(fonts.attrib.get(f"{{{W}}}ascii"), "Times New Roman")
                self.assertEqual(fonts.attrib.get(f"{{{W}}}hAnsi"), "Times New Roman")
                self.assertEqual(fonts.attrib.get(f"{{{W}}}eastAsia"), "黑体")
                self.assertEqual(run.find("w:rPr/w:b", {"w": W}).attrib.get(f"{{{W}}}val"), "0")

    def test_copies_external_hyperlink_relationships_used_by_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            styles = f"""<w:styles xmlns:w="{W}">
<w:style w:type="paragraph" w:default="1" w:styleId="a0"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="affa"><w:name w:val="毕设图题"/></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="af9"><w:name w:val="图表"/></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="affe"><w:name w:val="公式新标准"/></w:style>
</w:styles>"""
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:sectPr/></w:body></w:document>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:p><w:hyperlink r:id="rId99"><w:r><w:t>link</w:t></w:r></w:hyperlink></w:p></w:body></w:document>"""
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

    def test_updates_template_headers_without_changing_header_formatting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "front").mkdir()
            (root / "front" / "cover.tex").write_text(
                r"""
\heusetup{
  ctitle={中性原子量子计算体系的光子—原子量子接口仿真研究},
}
""",
                encoding="utf-8",
            )
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            header_rel_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:pPr><w:sectPr><w:headerReference w:type="default" r:id="rId1"/></w:sectPr></w:pPr></w:p>
<w:sectPr><w:headerReference w:type="default" r:id="rId2"/></w:sectPr>
</w:body></w:document>"""
            title_header = f"""<w:hdr xmlns:w="{W}"><w:p><w:pPr><w:pStyle w:val="1"/><w:pBdr><w:bottom w:val="thickThinSmallGap" w:sz="24" w:space="1" w:color="auto"/></w:pBdr></w:pPr><w:r><w:rPr><w:rFonts w:hint="eastAsia"/></w:rPr><w:t>论文题目</w:t></w:r></w:p></w:hdr>"""
            chapter_header = f"""<w:hdr xmlns:w="{W}"><w:p><w:pPr><w:pStyle w:val="Header"/><w:pBdr><w:bottom w:val="thickThinSmallGap" w:sz="24" w:space="1" w:color="auto"/></w:pBdr><w:rPr><w:sz w:val="21"/></w:rPr></w:pPr><w:r><w:rPr><w:sz w:val="21"/></w:rPr><w:t>第1章  绪论</w:t></w:r></w:p></w:hdr>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>1绪论</w:t></w:r></w:p>
<w:p><w:r><w:t>正文。</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>2方法</w:t></w:r></w:p>
</w:body></w:document>"""
            write_docx(
                template,
                {
                    "[Content_Types].xml": content_types,
                    "word/document.xml": template_document,
                    "word/header1.xml": title_header,
                    "word/header2.xml": chapter_header,
                    "word/_rels/document.xml.rels": (
                        f"<Relationships xmlns='{REL}'>"
                        f"<Relationship Id='rId1' Type='{header_rel_type}' Target='header1.xml'/>"
                        f"<Relationship Id='rId2' Type='{header_rel_type}' Target='header2.xml'/>"
                        "</Relationships>"
                    ),
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

            merge_with_template(template, intermediate, output, source_root=root)

            with zipfile.ZipFile(output) as archive:
                title_header_xml = archive.read("word/header1.xml").decode("utf-8")
                chapter_header_xml = archive.read("word/header2.xml").decode("utf-8")

            self.assertIn("中性原子量子计算体系的光子—原子量子接口仿真研究", title_header_xml)
            self.assertNotIn("论文题目", title_header_xml)
            self.assertIn('w:pStyle w:val="1"', title_header_xml)
            self.assertIn('w:bottom w:val="thickThinSmallGap"', title_header_xml)
            self.assertIn('w:rFonts w:hint="eastAsia"', title_header_xml)
            self.assertIn("STYLEREF", chapter_header_xml)
            self.assertIn("大标题", chapter_header_xml)
            self.assertIn('w:pStyle w:val="Header"', chapter_header_xml)
            self.assertIn('w:bottom w:val="thickThinSmallGap"', chapter_header_xml)
            self.assertIn('w:sz w:val="21"', chapter_header_xml)

    def test_moves_bibliography_entries_to_bibliography_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:sectPr/></w:body></w:document>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>1绪论</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>参考文献</w:t></w:r></w:p>
<w:p><w:r><w:t>DOCX_EXPORT_BIBLIOGRAPHY_MARKER</w:t></w:r></w:p>
<w:p><w:r><w:t>致谢正文。</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Bibliography"/></w:pPr><w:r><w:t>[1] First reference.</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Bibliography"/></w:pPr><w:r><w:t>[2] Second reference.</w:t></w:r></w:p>
</w:body></w:document>"""
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
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                },
            )

            merge_with_template(template, intermediate, output)

            with zipfile.ZipFile(output) as archive:
                document_xml = archive.read("word/document.xml")
            import xml.etree.ElementTree as ET

            root_xml = ET.fromstring(document_xml)
            paragraphs = ["".join(t.text or "" for t in p.findall(".//w:t", {"w": W})) for p in root_xml.findall(".//w:p", {"w": W})]
            compact = [text for text in paragraphs if text]
            self.assertNotIn("DOCX_EXPORT_BIBLIOGRAPHY_MARKER", compact)
            self.assertLess(compact.index("[1] First reference."), compact.index("致谢正文。"))
            self.assertLess(compact.index("[2] Second reference."), compact.index("致谢正文。"))

    def test_normalizes_numeric_citation_range_dash_in_superscript_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:sectPr/></w:body></w:document>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>1绪论</w:t></w:r></w:p>
<w:p><w:r><w:t>正文</w:t></w:r><w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>[1–3]</w:t></w:r><w:r><w:t>页码 69–73。</w:t></w:r></w:p>
</w:body></w:document>"""
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
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                },
            )

            merge_with_template(template, intermediate, output)

            with zipfile.ZipFile(output) as archive:
                document = archive.read("word/document.xml").decode("utf-8")
            self.assertIn("[1-3]", document)
            self.assertIn("69–73", document)

    def test_formats_images_tables_and_captions_with_template_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "body").mkdir()
            (root / "body" / "chapter1.aux").write_text(
                r"""
\newlabel{fig:system}{{1.1}{3}{系统示意图}{figure.1.1}{}}
\newlabel{tab:param}{{1.1}{4}{参数表}{table.1.1}{}}
""",
                encoding="utf-8",
            )
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:sectPr/></w:body></w:document>"""
            styles = f"""<w:styles xmlns:w="{W}">
<w:style w:type="paragraph" w:default="1" w:styleId="a0"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="affa"><w:name w:val="毕设图题"/></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="af9"><w:name w:val="图表"/></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="affe"><w:name w:val="公式新标准"/></w:style>
</w:styles>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>1绪论</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="BodyText"/></w:pPr><w:r><w:t>图 1.1 是正文引用 ABC 123。</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="CaptionedFigure"/></w:pPr><w:r><w:drawing/></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="ImageCaption"/></w:pPr><w:r><w:t>系统示意图。</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="TableCaption"/></w:pPr><w:r><w:t>参数表。</w:t></w:r></w:p>
<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/></w:tblPr><w:tr><w:tc><w:p><w:r><w:t>项目</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>数值</w:t></w:r></w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
<w:p><w:r><w:t>正文 ABC 123。</w:t></w:r></w:p>
</w:body></w:document>"""
            write_docx(
                template,
                {
                    "[Content_Types].xml": content_types,
                    "word/document.xml": template_document,
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                    "word/styles.xml": styles,
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

            merge_with_template(template, intermediate, output, source_root=root)

            with zipfile.ZipFile(output) as archive:
                document_xml = archive.read("word/document.xml")
                styles_xml = archive.read("word/styles.xml")
            import xml.etree.ElementTree as ET

            root_xml = ET.fromstring(document_xml)
            styles_root = ET.fromstring(styles_xml)
            paragraphs = root_xml.findall(".//w:p", {"w": W})
            image_paragraph = next(p for p in paragraphs if p.find(".//w:drawing", {"w": W}) is not None)
            image_style = image_paragraph.find("w:pPr/w:pStyle", {"w": W})
            image_spacing = image_paragraph.find("w:pPr/w:spacing", {"w": W})
            self.assertEqual(image_style.attrib[f"{{{W}}}val"], "aff6")
            self.assertEqual(image_spacing.attrib[f"{{{W}}}lineRule"], "auto")

            paragraph_texts = {
                "".join(t.text or "" for t in p.findall(".//w:t", {"w": W})): p
                for p in paragraphs
            }
            figure_caption = paragraph_texts["图1.1 系统示意图"]
            table_caption = paragraph_texts["表1.1 参数表"]
            figure_style = figure_caption.find("w:pPr/w:pStyle", {"w": W})
            table_caption_style = table_caption.find("w:pPr/w:pStyle", {"w": W})
            body_ref = paragraph_texts["图 1.1 是正文引用 ABC 123。"]
            body = paragraph_texts["正文 ABC 123。"]
            body_ref_style = body_ref.find("w:pPr/w:pStyle", {"w": W})
            body_style = body.find("w:pPr/w:pStyle", {"w": W})
            self.assertEqual(figure_style.attrib[f"{{{W}}}val"], "affa")
            self.assertEqual(table_caption_style.attrib[f"{{{W}}}val"], "af9")
            self.assertEqual(body_ref_style.attrib[f"{{{W}}}val"], "a0")
            self.assertEqual(body_style.attrib[f"{{{W}}}val"], "a0")
            self.assertNotIn("系统示意图。", paragraph_texts)
            self.assertNotIn("参数表。", paragraph_texts)

            figure_spacing = figure_caption.find("w:pPr/w:spacing", {"w": W})
            table_spacing = table_caption.find("w:pPr/w:spacing", {"w": W})
            figure_font = figure_caption.find("w:r/w:rPr/w:rFonts", {"w": W})
            table_font = table_caption.find("w:r/w:rPr/w:rFonts", {"w": W})
            self.assertEqual(figure_spacing.attrib[f"{{{W}}}line"], "440")
            self.assertEqual(figure_spacing.attrib[f"{{{W}}}lineRule"], "exact")
            self.assertEqual(table_spacing.attrib[f"{{{W}}}line"], "392")
            self.assertEqual(table_spacing.attrib[f"{{{W}}}lineRule"], "atLeast")
            self.assertEqual(figure_font.attrib[f"{{{W}}}eastAsia"], "宋体")
            self.assertEqual(figure_font.attrib[f"{{{W}}}ascii"], "Times New Roman")
            self.assertEqual(table_font.attrib[f"{{{W}}}eastAsia"], "宋体")
            self.assertEqual(table_font.attrib[f"{{{W}}}ascii"], "Times New Roman")

            body_spacing = body.find("w:pPr/w:spacing", {"w": W})
            body_indent = body.find("w:pPr/w:ind", {"w": W})
            body_jc = body.find("w:pPr/w:jc", {"w": W})
            body_font = body.find("w:r/w:rPr/w:rFonts", {"w": W})
            body_size = body.find("w:r/w:rPr/w:sz", {"w": W})
            body_size_cs = body.find("w:r/w:rPr/w:szCs", {"w": W})
            self.assertEqual(body_spacing.attrib[f"{{{W}}}line"], "440")
            self.assertEqual(body_spacing.attrib[f"{{{W}}}lineRule"], "exact")
            self.assertEqual(body_indent.attrib[f"{{{W}}}firstLine"], "480")
            self.assertEqual(body_indent.attrib[f"{{{W}}}firstLineChars"], "200")
            self.assertEqual(body_jc.attrib[f"{{{W}}}val"], "both")
            self.assertEqual(body_font.attrib[f"{{{W}}}eastAsia"], "宋体")
            self.assertEqual(body_font.attrib[f"{{{W}}}ascii"], "Times New Roman")
            self.assertEqual(body_size.attrib[f"{{{W}}}val"], "24")
            self.assertEqual(body_size_cs.attrib[f"{{{W}}}val"], "24")
            normalized_styles = {
                style.attrib[f"{{{W}}}styleId"]: style
                for style in styles_root.findall("w:style", {"w": W})
            }
            normal_fonts = normalized_styles["a0"].find("w:rPr/w:rFonts", {"w": W})
            caption_fonts = normalized_styles["affa"].find("w:rPr/w:rFonts", {"w": W})
            table_fonts = normalized_styles["af9"].find("w:rPr/w:rFonts", {"w": W})
            self.assertEqual(normal_fonts.attrib[f"{{{W}}}eastAsia"], "宋体")
            self.assertEqual(normal_fonts.attrib[f"{{{W}}}ascii"], "Times New Roman")
            self.assertEqual(caption_fonts.attrib[f"{{{W}}}eastAsia"], "宋体")
            self.assertEqual(table_fonts.attrib[f"{{{W}}}ascii"], "Times New Roman")

            table = root_xml.find(".//w:tbl", {"w": W})
            borders = table.find("w:tblPr/w:tblBorders", {"w": W})
            self.assertIsNotNone(borders)
            self.assertEqual(borders.find("w:top", {"w": W}).attrib[f"{{{W}}}sz"], "12")
            self.assertEqual(borders.find("w:bottom", {"w": W}).attrib[f"{{{W}}}sz"], "12")
            self.assertEqual(borders.find("w:insideH", {"w": W}).attrib[f"{{{W}}}sz"], "6")
            self.assertEqual(borders.find("w:insideV", {"w": W}).attrib[f"{{{W}}}sz"], "6")
            self.assertIsNone(borders.find("w:left", {"w": W}))
            self.assertIsNone(borders.find("w:right", {"w": W}))

    def test_formats_display_equations_and_resolves_equation_references_from_aux(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "body").mkdir()
            (root / "body" / "chapter2.aux").write_text(
                r"""
\@writefile{loe}{\contentsline {equation}{\numberline {2-1}}{13}{equation.2.1}\protected@file@percent }
\newlabel{eq:first}{{2-1}{13}{时间 bin 的定义}{equation.2.1}{}}
\@writefile{loe}{\contentsline {equation}{\numberline {2-2}}{13}{equation.2.2}\protected@file@percent }
\newlabel{eq:second}{{2-2}{13}{时间 bin 的定义}{equation.2.2}{}}
""",
                encoding="utf-8",
            )
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:sectPr/></w:body></w:document>"""
            formula = f"""<m:oMathPara><m:oMath><m:r><m:t>x=y</m:t></m:r></m:oMath></m:oMathPara>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}" xmlns:m="{M}"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>2方法</w:t></w:r></w:p>
<w:p>{formula}</w:p>
<w:p><w:r><w:t>由式</w:t></w:r><w:r><w:t> </w:t></w:r><w:r><w:t>[eq:first]</w:t></w:r><w:r><w:t> </w:t></w:r><w:r><w:t>可得，公式</w:t></w:r><w:r><w:t> </w:t></w:r><w:r><w:t>[eq:second]</w:t></w:r><w:r><w:t> </w:t></w:r><w:r><w:t>给出下一步。</w:t></w:r></w:p>
<w:p>{formula}</w:p>
</w:body></w:document>"""
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
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                },
            )

            merge_with_template(template, intermediate, output, source_root=root)

            with zipfile.ZipFile(output) as archive:
                document_xml = archive.read("word/document.xml")
            import xml.etree.ElementTree as ET

            root_xml = ET.fromstring(document_xml)
            text = "".join(item.text or "" for item in root_xml.findall(".//w:t", {"w": W}))
            document = document_xml.decode("utf-8")
            self.assertIn("由式（2-1）可得，公式（2-2）给出下一步。", text)
            self.assertNotIn("eq:first", text)
            self.assertIn("（2-1）", document)
            self.assertIn("（2-2）", document)
            self.assertIn('w:tab w:val="right"', document)
            self.assertIn('w:tab w:val="center"', document)

            paragraphs = root_xml.findall(".//w:p", {"w": W})
            formula_paragraph = next(p for p in paragraphs if "（2-1）" in "".join(t.text or "" for t in p.findall(".//w:t", {"w": W})))
            formula_style = formula_paragraph.find("w:pPr/w:pStyle", {"w": W})
            formula_indent = formula_paragraph.find("w:pPr/w:ind", {"w": W})
            formula_jc = formula_paragraph.find("w:pPr/w:jc", {"w": W})
            formula_spacing = formula_paragraph.find("w:pPr/w:spacing", {"w": W})
            number_run = formula_paragraph.findall("w:r", {"w": W})[-1]
            number_font = number_run.find("w:rPr/w:rFonts", {"w": W})
            number_size = number_run.find("w:rPr/w:sz", {"w": W})
            self.assertEqual(formula_style.attrib[f"{{{W}}}val"], "affe")
            self.assertEqual(formula_indent.attrib[f"{{{W}}}firstLine"], "0")
            self.assertEqual(formula_indent.attrib[f"{{{W}}}firstLineChars"], "0")
            self.assertEqual(formula_jc.attrib[f"{{{W}}}val"], "center")
            self.assertEqual(formula_spacing.attrib[f"{{{W}}}line"], "240")
            self.assertEqual(number_font.attrib[f"{{{W}}}ascii"], "Times New Roman")
            self.assertEqual(number_size.attrib[f"{{{W}}}val"], "24")

    def test_links_author_year_citations_to_bibliography_bookmarks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:sectPr/></w:body></w:document>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>1绪论</w:t></w:r></w:p>
<w:p><w:r><w:t>经典教材给出了标准表述。</w:t></w:r><w:r><w:t>(Nielsen and Chuang 2010)</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Bibliography"/></w:pPr><w:r><w:t>Nielsen, Michael A., and Isaac L. Chuang. 2010. Quantum Computation and Quantum Information.</w:t></w:r></w:p>
</w:body></w:document>"""
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
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                },
            )

            merge_with_template(template, intermediate, output)

            with zipfile.ZipFile(output) as archive:
                document = archive.read("word/document.xml")
            import xml.etree.ElementTree as ET

            doc = ET.fromstring(document)
            bookmarks = doc.findall(f".//{{{W}}}bookmarkStart")
            hyperlinks = doc.findall(f".//{{{W}}}hyperlink")
            self.assertTrue(any(item.attrib.get(f"{{{W}}}name", "").startswith("bibref_") for item in bookmarks))
            self.assertTrue(any(item.attrib.get(f"{{{W}}}anchor", "").startswith("bibref_") for item in hyperlinks))

    def test_builds_template_front_matter_from_cover_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "front").mkdir()
            (root / "front" / "cover.tex").write_text(
                r"""
\heusetup{
  statesecrets={公开},
  cnumber={no9527},
  natclassifiedindex={TM301.2},
  intclassifiedindex={62-5},
  ctitle={中性原子量子计算体系的光子—原子量子接口仿真研究},
  cxueke={工学},
  csubject={光学工程},
  caffil={物理与光电工程学院},
  cauthor={李铀},
  csupervisor={任晶\ 教授},
  cassosupervisor={刘哲\ 讲师},
  csubmitdate={2026年3月},
  etitle={Simulation of Photon--Atom Quantum Interfaces},
  eauthor={Li You},
  esupervisor={Professor Ren Jing},
  esubject={Optical Engineering},
  eaffil={School of Physics and Optoelectronic Engineering},
  esubmitdate={March, 2026},
  estudenttype={Master of Engineering},
  ckeywords={量子接口, 中性原子},
  ekeywords={Quantum Interface, Neutral Atom},
}

\begin{cabstract}
中文摘要正文。
\end{cabstract}

\begin{eabstract}
English abstract body.
\end{eabstract}
""",
                encoding="utf-8",
            )
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:r><w:t>分类号：                             密级：</w:t></w:r></w:p>
<w:p><w:r><w:t>U D C ：                             编号：</w:t></w:r></w:p>
<w:p><w:r><w:t>理(工)学硕士学位论文</w:t></w:r></w:p>
<w:p><w:r><w:t>论文题目</w:t></w:r></w:p>
<w:p/>
<w:p/>
<w:p/>
<w:p><w:r><w:t>Title of the Dissertation</w:t></w:r></w:p>
<w:p/>
<w:p/>
<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr><w:r><w:t>摘    要</w:t></w:r></w:p>
<w:p><w:r><w:t>摘要是学位论文的高度概括</w:t></w:r></w:p>
<w:p><w:r><w:t>关键词：宇宙</w:t></w:r></w:p>
<w:p><w:pPr><w:sectPr/></w:pPr></w:p>
<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr><w:r><w:t>Abstract</w:t></w:r></w:p>
<w:p><w:r><w:t>A precise translation of the Chinese version abstract.</w:t></w:r></w:p>
<w:p><w:r><w:t>Keywords: Universe</w:t></w:r></w:p>
<w:p><w:pPr><w:sectPr/></w:pPr></w:p>
<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr><w:r><w:t>目    录</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="11"/></w:pPr><w:r><w:t>第1章绪论1</w:t></w:r></w:p>
<w:p><w:pPr><w:sectPr/></w:pPr></w:p>
<w:p><w:pPr><w:pStyle w:val="aff8"/></w:pPr><w:r><w:t>绪论</w:t></w:r></w:p>
<w:p><w:pPr><w:sectPr><w:pgNumType w:start="1"/></w:sectPr></w:pPr></w:p>
<w:sectPr/>
</w:body></w:document>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:r><w:t>pandoc abstract should be skipped</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>1绪论</w:t></w:r></w:p>
<w:p><w:r><w:t>正文第一段。</w:t></w:r></w:p>
</w:body></w:document>"""
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
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                },
            )

            merge_with_template(template, intermediate, output, source_root=root)

            with zipfile.ZipFile(output) as archive:
                document = archive.read("word/document.xml").decode("utf-8")
                self.assertIn("分类号：TM301.2", document)
                self.assertIn("硕士研究生：李铀", document)
                self.assertIn("中文摘要正文。", document)
                self.assertIn("关键词：量子接口；中性原子", document)
                self.assertIn("第1章 绪论", document)
                self.assertNotIn("pandoc abstract should be skipped", document)
                self.assertNotIn("摘要是学位论文的高度概括", document)

    def test_uses_detailed_cover_tables_and_drops_duplicate_simple_cover(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "front").mkdir()
            (root / "front" / "cover.tex").write_text(
                r"""
\heusetup{
  statesecrets={公开},
  cnumber={no9527},
  natclassifiedindex={TM301.2},
  intclassifiedindex={62-5},
  ctitlecover={中性原子量子计算体系的\\光子—原子量子接口仿真研究},
  ctitle={中性原子量子计算体系的光子—原子量子接口仿真研究},
  cxueke={工学},
  csubject={光学工程},
  caffil={物理与光电工程学院},
  cauthor={李铀},
  csupervisor={任晶\ 教授},
  cassosupervisor={刘哲\ 讲师},
  csubmitdate={2026年3月},
  etitle={Simulation of Photon--Atom Quantum Interfaces},
  esubtitle={End-to-End Modeling},
  eauthor={Li You},
  esupervisor={Professor Ren Jing},
  esubject={Optical Engineering},
  eaffil={School of Physics and Optoelectronic Engineering},
  esubmitdate={March, 2026},
  estudenttype={Master of Engineering},
}
""",
                encoding="utf-8",
            )
            template = root / "template.docx"
            intermediate = root / "intermediate.docx"
            output = root / "output.docx"
            content_types = b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            simple_table = f"""<w:tbl><w:tr><w:tc><w:p><w:r><w:t>硕士研究生</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>：姓名</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"""
            classified_line = """<w:p><w:r><w:t>分类号：</w:t></w:r><w:r><w:rPr><w:u w:val="single"/></w:rPr><w:t xml:space="preserve">               </w:t></w:r><w:r><w:t xml:space="preserve">                               </w:t></w:r><w:r><w:t>密级：</w:t></w:r><w:r><w:rPr><w:u w:val="single"/></w:rPr><w:t xml:space="preserve">               </w:t></w:r></w:p>"""
            udc_line = """<w:p><w:r><w:t>U D C </w:t></w:r><w:r><w:t>：</w:t></w:r><w:r><w:rPr><w:u w:val="single"/></w:rPr><w:t xml:space="preserve">               </w:t></w:r><w:r><w:t xml:space="preserve">                               </w:t></w:r><w:r><w:t>编号：</w:t></w:r><w:r><w:rPr><w:u w:val="single"/></w:rPr><w:t xml:space="preserve">               </w:t></w:r></w:p>"""
            detailed_table = f"""<w:tbl>
<w:tr><w:tc><w:p><w:r><w:t>硕士研究生</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>：姓名  </w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>指导教师</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>：姓名  教授（副教授、讲师）</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>学位级别</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>：理（工）学硕士</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>学科、专业</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>：</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>所在单位</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>：物理与光电工程学院</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>论文提交日期</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>：20xx年x月</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>论文答辩日期</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>：20xx年x月</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>学位授予单位</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>：哈尔滨工程大学</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>"""
            english_table = f"""<w:tbl>
<w:tr><w:tc><w:p><w:r><w:t>Candidate:</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Full Name</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Supervisor:</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Prof.(Dr. ) Full Name</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Academic Degree Applied for:</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Master of Science</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Specialty:</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Optics/Physics</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Date of Submission:</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Dec. 20xx</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Date of Oral Examination:</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Mar. 20xx</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>University:</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Harbin Engineering University</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>"""
            template_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
<w:p><w:r><w:t>分类号：                             密级：</w:t></w:r></w:p>
<w:p><w:r><w:t>理学硕士学位论文</w:t></w:r></w:p>
<w:p><w:r><w:t>论文题目（25字以内）</w:t></w:r></w:p>
{simple_table}
<w:p><w:r><w:t>哈尔滨工程大学</w:t></w:r></w:p>
<w:p><w:r><w:t>20xx年xx月</w:t></w:r></w:p>
<w:p/>
{classified_line}
{udc_line}
<w:p><w:r><w:t>理(工)学硕士学位论文</w:t></w:r></w:p>
<w:p><w:r><w:t>论文题目</w:t></w:r></w:p>
{detailed_table}
<w:p><w:r><w:t>Classified Index:</w:t></w:r></w:p>
<w:p><w:r><w:t>U.D.C:</w:t></w:r></w:p>
<w:p><w:r><w:t>A Dissertation for the Degree of M. Sci</w:t></w:r></w:p>
<w:p><w:r><w:t>Title of the Dissertation</w:t></w:r></w:p>
<w:p/>
<w:p/>
<w:p/>
{english_table}
<w:p><w:r><w:t>哈尔滨工程大学</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr><w:r><w:t>摘    要</w:t></w:r></w:p>
<w:p><w:pPr><w:sectPr/></w:pPr></w:p>
<w:p><w:pPr><w:pStyle w:val="aff8"/></w:pPr><w:r><w:t>绪论</w:t></w:r></w:p>
<w:sectPr/>
</w:body></w:document>"""
            intermediate_document = f"""<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>1绪论</w:t></w:r></w:p></w:body></w:document>"""
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
                    "word/_rels/document.xml.rels": f"<Relationships xmlns='{REL}'/>",
                },
            )

            merge_with_template(template, intermediate, output, source_root=root)

            with zipfile.ZipFile(output) as archive:
                document_xml = archive.read("word/document.xml")
            import xml.etree.ElementTree as ET

            root_xml = ET.fromstring(document_xml)
            text = "".join(item.text or "" for item in root_xml.findall(".//w:t", {"w": W}))
            self.assertNotIn("论文题目（25字以内）", text)
            self.assertNotIn("理学硕士学位论文", text)
            self.assertNotIn("姓名", text)
            self.assertNotIn("20xx", text)
            self.assertIn("硕士研究生：李铀", text)
            self.assertIn("指导教师：任晶 教授", text)
            self.assertIn("学位级别：工学硕士", text)
            self.assertIn("学科、专业：光学工程", text)
            self.assertIn("Candidate:Li You", text)
            self.assertIn("Date of Oral Examination:", text)

            underlined_texts = [
                "".join(text_node.text or "" for text_node in run.findall("w:t", {"w": W}))
                for run in root_xml.findall(".//w:r", {"w": W})
                if run.find("w:rPr/w:u", {"w": W}) is not None
            ]
            self.assertTrue(any(value.startswith("TM301.2") and value.endswith(" ") for value in underlined_texts))
            self.assertTrue(any(value.startswith("公开") and value.endswith(" ") for value in underlined_texts))
            self.assertTrue(any(value.startswith("62-5") and value.endswith(" ") for value in underlined_texts))
            self.assertTrue(any(value.startswith("no9527") and value.endswith(" ") for value in underlined_texts))

            body = root_xml.find("w:body", {"w": W})
            self.assertIsNotNone(body)
            body_children = list(body)
            title_index = next(
                index
                for index, child in enumerate(body_children)
                if child.tag == f"{{{W}}}p" and "Simulation of Photon-Atom Quantum Interfaces" in "".join(
                    text_node.text or "" for text_node in child.findall(".//w:t", {"w": W})
                )
            )
            english_table_index = next(
                index for index, child in enumerate(body_children[title_index + 1 :], start=title_index + 1) if child.tag == f"{{{W}}}tbl"
            )
            blank_paragraphs = [
                child
                for child in body_children[title_index + 1 : english_table_index]
                if child.tag == f"{{{W}}}p" and not "".join(text_node.text or "" for text_node in child.findall(".//w:t", {"w": W})).strip()
            ]
            self.assertEqual(blank_paragraphs, [])
