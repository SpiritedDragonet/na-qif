from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compile_word_script_uses_project_docx_export():
    script = ROOT / "compile_word.ps1"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert 'Join-Path $scriptDir "thesis.docx"' in text
    assert 'python @docxArgs' in text
    assert '"-m", "docx_export"' in text
    assert '"--root", $scriptDir' in text
    assert '"--refresh-fields"' in text


def test_auto_compile_runs_word_export_after_tex_build_and_ignores_tmp_workspace():
    text = (ROOT / "auto_compile.ps1").read_text(encoding="utf-8")
    assert 'Join-Path $scriptDir "compile_word.ps1"' in text
    assert "Compile-WordThesis" in text
    assert text.index("function Compile-Thesis") < text.index("Compile-WordThesis")
    assert '".codex_tmp"' in text
