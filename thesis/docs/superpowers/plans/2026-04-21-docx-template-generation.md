# DOCX Template Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable command that converts the LaTeX thesis project into a template-based DOCX and validates the resulting Word package.

**Architecture:** The implementation uses Pandoc for LaTeX-to-DOCX content conversion, then applies OpenXML post-processing against `硕士毕业论文参考模板.docx`. The generator is a small Python package under `docx_export/`, uses only the Python standard library at runtime, and exposes `python -m docx_export` plus a `make word` target.

**Tech Stack:** Python standard library, Pandoc, OpenXML ZIP/XML package handling, optional Microsoft Word COM refresh on Windows, `unittest`.

---

### File Map

Create `docx_export/__init__.py` as the package marker.

Create `docx_export/__main__.py` as the module entry point for `python -m docx_export`.

Create `docx_export/config.py` to hold `ExportConfig`, path resolution, default filenames, and output directory policy.

Create `docx_export/preprocess.py` to build the temporary Pandoc workspace, copy required LaTeX assets, replace PDF figure references with PNG references, and normalize Pandoc-sensitive LaTeX commands.

Create `docx_export/pandoc_runner.py` to build and execute the Pandoc command, capture stdout/stderr, and return the intermediate DOCX path.

Create `docx_export/openxml.py` to provide Word package helpers for reading ZIP parts, writing ZIP parts, parsing XML, creating relationships, copying media, and preserving template parts.

Create `docx_export/postprocess.py` to merge the Pandoc body into the template package, apply paragraph style mapping, normalize figure and table captions, insert a TOC field placeholder, and prepare section/page-header repair hooks.

Create `docx_export/validate.py` to inspect the final DOCX and report missing media, unsupported media, LaTeX residue, missing TOC fields, missing OMML formulas, and caption/table/image count mismatches.

Create `docx_export/word_refresh.py` to optionally refresh fields through Word COM when `--refresh-fields` is requested and `win32com` is available.

Create `tests/__init__.py`, `tests/docx_export/__init__.py`, `tests/docx_export/test_config.py`, `tests/docx_export/test_preprocess.py`, `tests/docx_export/test_openxml.py`, `tests/docx_export/test_validate.py`, and `tests/docx_export/test_cli.py` for standard-library `unittest` coverage.

Modify `Makefile` to add a `word` target that runs `python -m docx_export --root . --output thesis_word_final.docx`.

Modify `.gitignore` only if generated DOCX or temporary output paths are still unignored after checking the current ignore rules.

### Task 1: Package Skeleton And Configuration

**Files:**
- Create: `docx_export/__init__.py`
- Create: `docx_export/__main__.py`
- Create: `docx_export/config.py`
- Create: `tests/__init__.py`
- Create: `tests/docx_export/__init__.py`
- Create: `tests/docx_export/test_config.py`

- [ ] **Step 1: Write the failing config tests**

Create `tests/docx_export/test_config.py` with these tests.

```python
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
```

- [ ] **Step 2: Run the config tests and confirm failure**

Run: `python -m unittest tests.docx_export.test_config -v`

Expected: import failure for `docx_export.config`.

- [ ] **Step 3: Implement the minimal configuration objects**

Create `docx_export/config.py`.

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportConfig:
    root: Path
    main_tex: Path
    template_docx: Path
    output_docx: Path
    work_dir: Path
    refresh_fields: bool = False

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        output: str | Path | None = None,
        template: str | Path | None = None,
        refresh_fields: bool = False,
    ) -> "ExportConfig":
        root_path = Path(root).resolve()
        main_tex = root_path / "thesis.tex"
        template_docx = Path(template).resolve() if template else root_path / "硕士毕业论文参考模板.docx"
        output_docx = Path(output).resolve() if output else root_path / "thesis_word_final.docx"
        if not main_tex.exists():
            raise FileNotFoundError(f"Missing thesis.tex: {main_tex}")
        if not template_docx.exists():
            raise FileNotFoundError(f"Missing template DOCX: {template_docx}")
        return cls(
            root=root_path,
            main_tex=main_tex,
            template_docx=template_docx,
            output_docx=output_docx,
            work_dir=root_path / ".codex_tmp" / "docx_export",
            refresh_fields=refresh_fields,
        )
```

Create `docx_export/__init__.py` with a short package docstring.

Create `tests/__init__.py` and `tests/docx_export/__init__.py` as empty files so task-specific `unittest` module paths import reliably.

Create `docx_export/__main__.py` with a placeholder CLI that parses `--root`, `--output`, `--template`, and `--refresh-fields`, then prints the resolved config. The real pipeline is wired in Task 8.

- [ ] **Step 4: Run the config tests and confirm pass**

Run: `python -m unittest tests.docx_export.test_config -v`

Expected: both tests pass.

- [ ] **Step 5: Commit the skeleton**

Run:

```powershell
git -C G:\BProj\Quantum_simulation add -- thesis/docx_export thesis/tests/__init__.py thesis/tests/docx_export/__init__.py thesis/tests/docx_export/test_config.py
git -C G:\BProj\Quantum_simulation commit -m "Add docx export configuration" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: commit contains only the package skeleton and config test.

### Task 2: LaTeX Workspace Preprocessing

**Files:**
- Create: `docx_export/preprocess.py`
- Create: `tests/docx_export/test_preprocess.py`

- [ ] **Step 1: Write tests for figure replacement and macro cleanup**

Create `tests/docx_export/test_preprocess.py`.

```python
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
```

- [ ] **Step 2: Run the preprocess tests and confirm failure**

Run: `python -m unittest tests.docx_export.test_preprocess -v`

Expected: import failure for `docx_export.preprocess`.

- [ ] **Step 3: Implement workspace creation**

Create `docx_export/preprocess.py`. Copy required top-level files with suffixes `.tex`, `.bib`, `.cls`, `.sty`, `.cfg`, `.bst`, `.ist`; copy `front/`, `body/`, `back/`, and `figures/` when present. Use `shutil.copytree(..., dirs_exist_ok=True)` and clean only `config.work_dir` after checking that it is under `config.root / ".codex_tmp"`.

- [ ] **Step 4: Implement LaTeX text normalization**

Add a pure function `normalize_latex_for_pandoc(text: str, root: Path) -> str`. It should replace figure references matching `{... .pdf}` with `{... .png}` when the PNG exists under the project root or temporary root. It should replace `\rm` with `\mathrm`, remove `\allowbreak`, and simplify `\Bigl`, `\Bigr`, `\bigl`, and `\bigr` to empty strings.

- [ ] **Step 5: Apply normalization to copied `.tex` files**

Walk the temporary workspace and rewrite every `.tex` file with UTF-8 encoding. Preserve file names and relative paths.

- [ ] **Step 6: Run preprocess tests and confirm pass**

Run: `python -m unittest tests.docx_export.test_preprocess -v`

Expected: both tests pass.

- [ ] **Step 7: Commit preprocessing**

Run:

```powershell
git -C G:\BProj\Quantum_simulation add -- thesis/docx_export/preprocess.py thesis/tests/docx_export/test_preprocess.py
git -C G:\BProj\Quantum_simulation commit -m "Add latex preprocessing for docx export" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: commit contains only preprocessing code and tests.

### Task 3: Pandoc Runner

**Files:**
- Create: `docx_export/pandoc_runner.py`
- Create: `tests/docx_export/test_pandoc_runner.py`

- [ ] **Step 1: Write command-building tests**

Create `tests/docx_export/test_pandoc_runner.py`.

```python
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

            command = build_pandoc_command(config, workspace, workspace / "intermediate.docx")

            self.assertEqual(command[0], "pandoc")
            self.assertIn("thesis.tex", command)
            self.assertIn("--from=latex", command)
            self.assertIn("--to=docx", command)
            self.assertTrue(any(item.startswith("--resource-path=") for item in command))
            self.assertTrue(any(item.startswith("--output=") for item in command))
```

- [ ] **Step 2: Run the Pandoc tests and confirm failure**

Run: `python -m unittest tests.docx_export.test_pandoc_runner -v`

Expected: import failure for `docx_export.pandoc_runner`.

- [ ] **Step 3: Implement command construction**

Create `docx_export/pandoc_runner.py` with `build_pandoc_command(config, workspace, output_docx)`. The command should include `thesis.tex`, `--from=latex`, `--to=docx`, `--resource-path=.;figures;..;../figures`, `--citeproc` when `reference.bib` exists, `--bibliography=reference.bib` when present, `--number-sections`, and `--output=<path>`.

- [ ] **Step 4: Implement Pandoc execution**

Add `run_pandoc(config, workspace) -> Path`. Use `subprocess.run(command, cwd=workspace, text=True, capture_output=True)`. On non-zero exit, raise `RuntimeError` with stdout and stderr summaries. Return `workspace / "intermediate.docx"` on success.

- [ ] **Step 5: Run tests and confirm pass**

Run: `python -m unittest tests.docx_export.test_pandoc_runner -v`

Expected: command-building test passes.

- [ ] **Step 6: Smoke-test command availability**

Run: `where.exe pandoc`

Expected: output includes `C:\Program Files\Pandoc\pandoc.exe`.

- [ ] **Step 7: Commit Pandoc runner**

Run:

```powershell
git -C G:\BProj\Quantum_simulation add -- thesis/docx_export/pandoc_runner.py thesis/tests/docx_export/test_pandoc_runner.py
git -C G:\BProj\Quantum_simulation commit -m "Add pandoc runner for docx export" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: commit contains only the Pandoc runner and tests.

### Task 4: OpenXML Package Helpers

**Files:**
- Create: `docx_export/openxml.py`
- Create: `tests/docx_export/test_openxml.py`

- [ ] **Step 1: Write tests for package part reading and writing**

Create `tests/docx_export/test_openxml.py`.

```python
import tempfile
import unittest
import zipfile
from pathlib import Path

from docx_export.openxml import DocxPackage


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
```

- [ ] **Step 2: Run OpenXML tests and confirm failure**

Run: `python -m unittest tests.docx_export.test_openxml -v`

Expected: import failure for `docx_export.openxml`.

- [ ] **Step 3: Implement `DocxPackage`**

Create a dataclass with `parts: dict[str, bytes]`, `read(path)`, `write(path)`, `xml_part(name)`, `set_xml_part(name, element)`, and `next_relationship_id(rels_xml)`. Use `zipfile.ZipFile`, `xml.etree.ElementTree`, and deterministic sorted writes where safe. Preserve binary parts unchanged.

- [ ] **Step 4: Add XML namespace constants**

Add constants for WordprocessingML, relationships, drawing, and package relationships. Register namespaces with `ElementTree.register_namespace` so rewritten XML remains readable.

- [ ] **Step 5: Run OpenXML tests and confirm pass**

Run: `python -m unittest tests.docx_export.test_openxml -v`

Expected: both tests pass.

- [ ] **Step 6: Commit OpenXML helpers**

Run:

```powershell
git -C G:\BProj\Quantum_simulation add -- thesis/docx_export/openxml.py thesis/tests/docx_export/test_openxml.py
git -C G:\BProj\Quantum_simulation commit -m "Add openxml package helpers" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: commit contains only OpenXML helpers and tests.

### Task 5: DOCX Validation

**Files:**
- Create: `docx_export/validate.py`
- Create: `tests/docx_export/test_validate.py`

- [ ] **Step 1: Write validation tests for LaTeX residue and media checks**

Create `tests/docx_export/test_validate.py`.

```python
import tempfile
import unittest
import zipfile
from pathlib import Path

from docx_export.validate import validate_docx


class ValidateDocxTests(unittest.TestCase):
    def test_detects_latex_residue(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "bad.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr("word/document.xml", "<w:t>\\rm x</w:t>")

            report = validate_docx(docx)

            self.assertFalse(report.ok)
            self.assertIn("LaTeX residue", "\n".join(report.errors))

    def test_detects_unsupported_pdf_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "bad.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr("word/document.xml", "<w:document/>")
                archive.writestr("word/media/figure.pdf", b"%PDF")

            report = validate_docx(docx)

            self.assertFalse(report.ok)
            self.assertIn("Unsupported media", "\n".join(report.errors))
```

- [ ] **Step 2: Run validation tests and confirm failure**

Run: `python -m unittest tests.docx_export.test_validate -v`

Expected: import failure for `docx_export.validate`.

- [ ] **Step 3: Implement validation report**

Create `ValidationReport` with `errors`, `warnings`, and `ok` property. Implement `validate_docx(path) -> ValidationReport`.

- [ ] **Step 4: Implement first validation rules**

Read all XML parts as UTF-8 with replacement. Flag `\rm`, `\Bigl`, `\Bigr`, `\allowbreak`, and raw `\includegraphics`. Flag `word/media/*.pdf` and `word/media/*.eps`. Flag a missing `word/document.xml`. Count `TOC`, `<m:oMath`, `图`, `表`, `word/media`, and `word/_rels/document.xml.rels` references, reporting warnings for suspicious mismatches.

- [ ] **Step 5: Run validation tests and confirm pass**

Run: `python -m unittest tests.docx_export.test_validate -v`

Expected: validation tests pass.

- [ ] **Step 6: Commit validation**

Run:

```powershell
git -C G:\BProj\Quantum_simulation add -- thesis/docx_export/validate.py thesis/tests/docx_export/test_validate.py
git -C G:\BProj\Quantum_simulation commit -m "Add docx validation checks" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: commit contains only validation code and tests.

### Task 6: Template Body Merge And Basic Styling

**Files:**
- Create: `docx_export/postprocess.py`
- Modify: `docx_export/openxml.py`
- Create: `tests/docx_export/test_postprocess.py`

- [ ] **Step 1: Write a minimal merge test**

Create `tests/docx_export/test_postprocess.py`. Build two tiny DOCX packages in temporary files: a template with `word/styles.xml`, `word/settings.xml`, `word/document.xml` containing an empty body, and a Pandoc document with `word/document.xml` containing one heading and one paragraph. Assert that `merge_with_template(template, intermediate, output)` writes a final package whose `word/document.xml` contains the Pandoc paragraph and whose `word/styles.xml` equals the template styles part.

- [ ] **Step 2: Run postprocess tests and confirm failure**

Run: `python -m unittest tests.docx_export.test_postprocess -v`

Expected: import failure for `docx_export.postprocess`.

- [ ] **Step 3: Implement `merge_with_template`**

Read both packages. Copy the template package, replace the body content in template `word/document.xml` with the body children from the Pandoc `word/document.xml`, and preserve the final template `sectPr` as the final body section properties. Write the output DOCX.

- [ ] **Step 4: Preserve and remap media**

Copy all Pandoc `word/media/*` parts into the final package. Merge `word/_rels/document.xml.rels` image relationships from Pandoc into the template relationship file with fresh `rId` values. Update copied drawing relationship IDs in `word/document.xml` to the new IDs.

- [ ] **Step 5: Add paragraph style mapping hooks**

Implement functions that detect headings by Pandoc style IDs or outline levels, then assign template style IDs. Map chapter-like paragraphs to `aff8`, heading level 2 to `2`, heading level 3 to `3`, ordinary paragraphs to `a0` or direct body formatting, figure captions to centered caption formatting, table captions to centered caption formatting, and bibliography paragraphs to the selected reference style.

- [ ] **Step 6: Run postprocess tests and validation tests**

Run: `python -m unittest tests.docx_export.test_postprocess tests.docx_export.test_openxml tests.docx_export.test_validate -v`

Expected: all listed tests pass.

- [ ] **Step 7: Commit merge and styling**

Run:

```powershell
git -C G:\BProj\Quantum_simulation add -- thesis/docx_export/postprocess.py thesis/docx_export/openxml.py thesis/tests/docx_export/test_postprocess.py
git -C G:\BProj\Quantum_simulation commit -m "Merge pandoc docx into thesis template" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: commit contains template merge and basic styling logic.

### Task 7: Optional Word Field Refresh

**Files:**
- Create: `docx_export/word_refresh.py`
- Create: `tests/docx_export/test_word_refresh.py`

- [ ] **Step 1: Write tests for graceful no-op behavior**

Create `tests/docx_export/test_word_refresh.py`. Test that `refresh_fields(path, enabled=False)` returns a status object with `attempted=False`. Test that `refresh_fields(path, enabled=True)` returns a clear skipped or failed status when `win32com` import fails; use `unittest.mock.patch.dict("sys.modules", {"win32com": None})`.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m unittest tests.docx_export.test_word_refresh -v`

Expected: import failure for `docx_export.word_refresh`.

- [ ] **Step 3: Implement refresh status**

Create `RefreshStatus` with `attempted`, `succeeded`, and `message`. Implement `refresh_fields(path, enabled)`.

- [ ] **Step 4: Implement Windows COM refresh**

When enabled and `win32com.client` imports successfully, open Word in background, open the DOCX, update fields in the document body, headers, footers, and tables of contents, save, and close. Always quit Word in `finally`. Return a status message that states whether fields were refreshed.

- [ ] **Step 5: Run refresh tests**

Run: `python -m unittest tests.docx_export.test_word_refresh -v`

Expected: tests pass without launching Word.

- [ ] **Step 6: Commit refresh module**

Run:

```powershell
git -C G:\BProj\Quantum_simulation add -- thesis/docx_export/word_refresh.py thesis/tests/docx_export/test_word_refresh.py
git -C G:\BProj\Quantum_simulation commit -m "Add optional Word field refresh" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: commit contains only optional field refresh code and tests.

### Task 8: CLI Integration And Make Target

**Files:**
- Modify: `docx_export/__main__.py`
- Modify: `Makefile`
- Create: `tests/docx_export/test_cli.py`

- [ ] **Step 1: Write CLI tests with mocked pipeline functions**

Create `tests/docx_export/test_cli.py`. Use `unittest.mock` to patch `prepare_pandoc_workspace`, `run_pandoc`, `merge_with_template`, `validate_docx`, and `refresh_fields`. Assert that `main(["--root", tmp])` calls the functions in order and returns `0` when validation passes. Assert that it returns `2` when validation returns errors.

- [ ] **Step 2: Run CLI tests and confirm failure**

Run: `python -m unittest tests.docx_export.test_cli -v`

Expected: tests fail because the CLI still prints only the config.

- [ ] **Step 3: Wire the full pipeline**

Update `docx_export/__main__.py` so `main(argv=None)` resolves config, prepares the workspace, runs Pandoc, merges the intermediate DOCX into the template, validates the final DOCX, optionally refreshes Word fields, prints a compact report, and returns a shell exit code.

- [ ] **Step 4: Add validate-only mode**

Add `--validate-only` to the CLI. When this flag is present, resolve the output DOCX path, run `validate_docx` against it, print the same validation report, and skip preprocessing, Pandoc, merge, and field refresh. Cover the branch in `tests/docx_export/test_cli.py`.

- [ ] **Step 5: Add the Makefile target**

Modify `Makefile` with a `word` target.

```make
word:
	python -m docx_export --root . --output thesis_word_final.docx
```

Add `word` to `.PHONY`.

- [ ] **Step 6: Run CLI tests**

Run: `python -m unittest tests.docx_export.test_cli -v`

Expected: CLI tests pass.

- [ ] **Step 7: Run the full unit test suite**

Run: `python -m unittest discover -s tests -p "test_*.py" -v`

Expected: all DOCX export tests pass.

- [ ] **Step 8: Commit CLI integration**

Run:

```powershell
git -C G:\BProj\Quantum_simulation add -- thesis/docx_export thesis/tests/docx_export/test_cli.py thesis/Makefile
git -C G:\BProj\Quantum_simulation commit -m "Wire docx export cli" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: commit contains CLI and Makefile integration.

### Task 9: End-To-End Export And XML Verification

**Files:**
- Modify only if needed: `docx_export/*.py`
- Output: `thesis_word_final.docx`

- [ ] **Step 1: Run the unit test suite**

Run: `python -m unittest discover -s tests -p "test_*.py" -v`

Expected: all DOCX export tests pass.

- [ ] **Step 2: Run the actual export**

Run: `python -m docx_export --root . --output thesis_word_final.docx`

Expected: Pandoc creates an intermediate DOCX, postprocessing writes `thesis_word_final.docx`, and validation reports no errors. Warnings are acceptable for fields that need Word refresh.

- [ ] **Step 3: Inspect the resulting DOCX package**

Run:

```powershell
python -c "import zipfile, pathlib; p=pathlib.Path('thesis_word_final.docx'); z=zipfile.ZipFile(p); print(p.exists(), p.stat().st_size); print(len([n for n in z.namelist() if n.startswith('word/media/')]))"
```

Expected: output file exists, file size is non-trivial, and media count is greater than zero.

- [ ] **Step 4: Run validator directly**

Run: `python -m docx_export --root . --output thesis_word_final.docx --validate-only`

Expected: validation completes with zero errors. If `--validate-only` has not been implemented, add it to the CLI in this task and cover it with a CLI test.

- [ ] **Step 5: Optionally refresh Word fields**

Run: `python -m docx_export --root . --output thesis_word_final.docx --refresh-fields`

Expected: on a machine with Word COM available, fields refresh and the command exits with status 0. On a machine without Word COM, the command reports a skipped refresh and keeps the generated DOCX.

- [ ] **Step 6: Commit end-to-end fixes**

Run:

```powershell
git -C G:\BProj\Quantum_simulation add -- thesis/docx_export thesis/tests/docx_export thesis/Makefile
git -C G:\BProj\Quantum_simulation commit -m "Stabilize docx export pipeline" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: final implementation commit contains only source, tests, and Makefile updates. Generated `thesis_word_final.docx` should remain untracked unless the user explicitly asks to commit output files.

### Task 10: Visual Spot Check And Follow-Up List

**Files:**
- Modify: `docs/superpowers/plans/2026-04-21-docx-template-generation.md` only if implementation findings require plan correction.
- Output: browser companion screen or concise terminal report.

- [ ] **Step 1: Produce representative page previews if a PDF conversion path is available**

Run a local DOCX-to-PDF or Word export path only when available. If conversion is unavailable, inspect the DOCX XML and open the DOCX manually in Word.

Expected: representative pages include cover, abstract, TOC, body, equation, and figure/table pages.

- [ ] **Step 2: Use browser companion for side-by-side comparison when visual assets exist**

Create a companion screen showing template screenshots and generated screenshots for the same page classes. Ask the user to judge spacing, title placement, caption placement, and overall template fidelity.

Expected: user can identify the highest-priority visual mismatch.

- [ ] **Step 3: Record follow-up issues**

Document unresolved high-fidelity tasks such as exact cover metadata placement, dynamic `STYLEREF` headers, cross-reference fields, complex table continuation labels, and formula number tab stops.

Expected: follow-up list is concrete enough to become the next implementation plan.

- [ ] **Step 4: Final verification before handoff**

Run:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m docx_export --root . --output thesis_word_final.docx
```

Expected: unit tests pass and the DOCX export command exits successfully.
