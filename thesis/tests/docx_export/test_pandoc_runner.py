import tempfile
import unittest
from pathlib import Path

from docx_export.config import ExportConfig
from docx_export.pandoc_runner import build_pandoc_command


class PandocRunnerTests(unittest.TestCase):
    def test_builds_expected_pandoc_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "thesis.tex").write_text("% thesis", encoding="utf-8")
            (root / "reference.bib").write_text("", encoding="utf-8")
            (root / "硕士毕业论文参考模板.docx").write_bytes(b"docx")
            config = ExportConfig.from_root(root)
            workspace = config.work_dir
            workspace.mkdir(parents=True)
            (workspace / "reference.bib").write_text("", encoding="utf-8")

            command = build_pandoc_command(config, workspace, workspace / "intermediate.docx")

            self.assertEqual(command[0], "pandoc")
            self.assertIn("thesis.tex", command)
            self.assertIn("--from=latex", command)
            self.assertIn("--to=docx", command)
            self.assertIn("--citeproc", command)
            self.assertTrue(any(item.startswith("--csl=") and item.endswith("gbt7714-numeric.csl") for item in command))
            self.assertTrue(any(item.startswith("--resource-path=") for item in command))
            self.assertTrue(any(item.startswith("--output=") for item in command))
