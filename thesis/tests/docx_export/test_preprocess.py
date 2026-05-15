import tempfile
import unittest
from pathlib import Path

from docx_export.config import ExportConfig
from docx_export.preprocess import prepare_pandoc_workspace


class PreprocessTests(unittest.TestCase):
    def test_rewrites_pdf_figure_to_png_when_png_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "figures").mkdir()
            (root / "figures" / "diagram.png").write_bytes(b"png")
            (root / "thesis.tex").write_text(
                r"\\includegraphics[width=0.8\\textwidth]{figures/diagram.pdf}",
                encoding="utf-8",
            )
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"docx")
            config = ExportConfig.from_root(root)

            workspace = prepare_pandoc_workspace(config)

            converted = (workspace / "thesis.tex").read_text(encoding="utf-8")
            self.assertIn("{figures/diagram.png}", converted)

    def test_removes_pandoc_sensitive_latex_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "thesis.tex").write_text(r"$\\rm x \\allowbreak \\Bigl( y \\Bigr)$", encoding="utf-8")
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"docx")
            config = ExportConfig.from_root(root)

            workspace = prepare_pandoc_workspace(config)

            converted = (workspace / "thesis.tex").read_text(encoding="utf-8")
            self.assertNotIn(r"\\rm", converted)
            self.assertNotIn(r"\\allowbreak", converted)
            self.assertNotIn(r"\\Bigl", converted)

    def test_rewrites_bare_pdf_name_when_png_exists_in_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "figures").mkdir()
            (root / "figures" / "diagram.png").write_bytes(b"png")
            (root / "thesis.tex").write_text(r"\\includegraphics{diagram.pdf}", encoding="utf-8")
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"docx")
            config = ExportConfig.from_root(root)

            workspace = prepare_pandoc_workspace(config)

            converted = (workspace / "thesis.tex").read_text(encoding="utf-8")
            self.assertIn("{diagram.png}", converted)

    def test_removes_heu_trailing_english_heading_titles_before_pandoc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "body").mkdir()
            (root / "thesis.tex").write_text(r"\input{body/chapter1.tex}", encoding="utf-8")
            (root / "body" / "chapter1.tex").write_text(
                "\n".join(
                    [
                        r"\chapter{绪论}[Introduction]",
                        r"\section{量子计算简介}[Introduction to quantum computing]",
                        r"\subsection{量子计算基本概念}[Basic Concepts]",
                        r"\subsubsection{误差来源}[Error Sources]",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"docx")
            config = ExportConfig.from_root(root)

            workspace = prepare_pandoc_workspace(config)

            converted = (workspace / "body" / "chapter1.tex").read_text(encoding="utf-8")
            self.assertIn(r"\chapter{绪论}", converted)
            self.assertIn(r"\section{量子计算简介}", converted)
            self.assertIn(r"\subsection{量子计算基本概念}", converted)
            self.assertIn(r"\subsubsection{误差来源}", converted)
            self.assertNotIn("[Introduction]", converted)
            self.assertNotIn("[Introduction to quantum computing]", converted)

    def test_replaces_bibliography_command_with_docx_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "thesis.tex").write_text(
                "\n".join(
                    [
                        r"\bibliographystyle{gbt7714-numerical}",
                        r"\bibliography{reference}",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "reference.bib").write_text("", encoding="utf-8")
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"docx")
            config = ExportConfig.from_root(root)

            workspace = prepare_pandoc_workspace(config)

            converted = (workspace / "thesis.tex").read_text(encoding="utf-8")
            self.assertIn(r"\section*{参考文献}", converted)
            self.assertIn("DOCX_EXPORT_BIBLIOGRAPHY_MARKER", converted)
            self.assertNotIn(r"\bibliographystyle", converted)
            self.assertNotIn(r"\bibliography{reference}", converted)
