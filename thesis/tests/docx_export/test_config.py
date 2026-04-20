import tempfile
import unittest
from pathlib import Path

from docx_export.config import ExportConfig


class ExportConfigTests(unittest.TestCase):
    def test_defaults_resolve_project_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "thesis.tex").write_text("% thesis", encoding="utf-8")
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"placeholder")

            config = ExportConfig.from_root(root)

            self.assertEqual(config.root, root.resolve())
            self.assertEqual(config.main_tex, root.resolve() / "thesis.tex")
            self.assertEqual(config.template_docx, root.resolve() / "硕士毕业论文参考模板.docx")
            self.assertEqual(config.output_docx, root.resolve() / "thesis_word_final.docx")
            self.assertEqual(config.work_dir, root.resolve() / ".codex_tmp" / "docx_export")

    def test_missing_main_tex_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"placeholder")

            with self.assertRaisesRegex(FileNotFoundError, "thesis.tex"):
                ExportConfig.from_root(root)

