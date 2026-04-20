from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RefreshStatus:
    attempted: bool
    succeeded: bool
    message: str


def refresh_fields(path: str | Path, enabled: bool = False) -> RefreshStatus:
    if not enabled:
        return RefreshStatus(attempted=False, succeeded=False, message="Word field refresh disabled.")
    try:
        import win32com.client  # type: ignore[import-not-found]
    except Exception as exc:
        return RefreshStatus(attempted=True, succeeded=False, message=f"Word COM refresh skipped: {exc}")

    word = None
    document = None
    try:
        docx_path = Path(path).resolve()
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        document = word.Documents.Open(str(docx_path))
        document.Fields.Update()
        for toc in document.TablesOfContents:
            toc.Update()
        for section in document.Sections:
            for header in section.Headers:
                header.Range.Fields.Update()
            for footer in section.Footers:
                footer.Range.Fields.Update()
        document.Save()
        return RefreshStatus(attempted=True, succeeded=True, message="Word fields refreshed.")
    except Exception as exc:
        return RefreshStatus(attempted=True, succeeded=False, message=f"Word COM refresh failed: {exc}")
    finally:
        if document is not None:
            try:
                document.Close(False)
            except Exception:
                pass
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass

