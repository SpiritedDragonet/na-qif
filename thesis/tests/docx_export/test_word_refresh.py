import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docx_export.word_refresh import refresh_fields


class WordRefreshTests(unittest.TestCase):
    def test_disabled_refresh_is_noop(self):
        status = refresh_fields(Path("missing.docx"), enabled=False)

        self.assertFalse(status.attempted)
        self.assertFalse(status.succeeded)

    def test_missing_win32com_reports_skipped_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "out.docx"
            docx.write_bytes(b"docx")
            with patch.dict(sys.modules, {"win32com": None}):
                status = refresh_fields(docx, enabled=True)

        self.assertTrue(status.attempted)
        self.assertFalse(status.succeeded)
        self.assertIn("skipped", status.message)

