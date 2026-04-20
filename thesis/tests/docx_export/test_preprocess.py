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

