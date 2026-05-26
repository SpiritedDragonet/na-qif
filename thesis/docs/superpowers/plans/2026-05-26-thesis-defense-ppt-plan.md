# Thesis Defense PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ten-minute Chinese thesis defense PPTX for《中性原子量子计算体系的光子--原子量子接口仿真研究》with rendered PNG QA and a short QA report.

**Architecture:** Use the approved story design in `docs/superpowers/specs/2026-05-26-thesis-defense-ppt-design.md` as the source of truth. Extract only the thesis figures needed for the 11-slide defense story, build an editable PPTX with `python-pptx`, then render or structurally inspect each slide and fix visible layout problems. Keep generated deliverables under an ignored local output directory so thesis source files are not disturbed.

**Tech Stack:** PyMuPDF for PDF/page/image conversion when needed, Pillow for image inspection/contact sheets, python-pptx for PPTX authoring, zipfile/python-pptx reopen checks for package validation, optional LibreOffice/PowerPoint-free rendering only if already available.

---

### Task 1: Confirm Source Assets and Tooling

**Files:**
- Read: `thesis.pdf`
- Read: `front/cover.tex`
- Read: `body/chapter1.tex`
- Read: `body/chapter2.tex`
- Read: `body/chapter3.tex`
- Read: `body/chapter4.tex`
- Read: `back/conclusion.tex`
- Read: `figures/*`
- Create: `outputs/defense_ppt/asset_manifest.md`

- [ ] **Step 1: Check Python package availability**

Run:

```powershell
python -c "import pptx, PIL, fitz; print('python-pptx/Pillow/PyMuPDF available')"
```

Expected: command prints package availability. If a package is missing, either use an already available alternative or request approval before installing the minimum package.

- [ ] **Step 2: Inspect selected figure dimensions**

Run a `python -c` command to print image dimensions and PDF page counts for candidate figures including `fig_ch1_modular_network`, `fig_ch1_scheme_and_link`, `fig_ch2_timebin_mps_tebd`, `fig_sim_pipeline`, `fig_window_tradeoff`, `fig_hom`, `fig_bsm_patterns`, `fig_distance`, `fig_qfc_noise`, and `fig_error_budget`.

Expected: selected figures are readable and have enough resolution for 16:9 slides.

- [ ] **Step 3: Write the asset manifest**

Create `outputs/defense_ppt/asset_manifest.md` with each selected figure, intended slide, source path, conversion status, and notes about whether it is used as a full figure, crop, or editable placeholder for later Morph refinement.

Expected: manifest records only figures used in the deck or planned as backup evidence.

### Task 2: Draft Slide Content and Speaker Notes

**Files:**
- Read: `docs/superpowers/specs/2026-05-26-thesis-defense-ppt-design.md`
- Create: `outputs/defense_ppt/slide_outline.md`

- [ ] **Step 1: Write the 11-slide outline**

Create `outputs/defense_ppt/slide_outline.md` with slide title, one-sentence claim, visual asset, 2 to 4 concise page bullets, and speaker-note cue for each slide.

Expected: no slide contains internal strategy wording such as “让评委理解”“避免老师误解”“大同行会觉得”.

- [ ] **Step 2: Add backup slide notes**

Add backup slide candidates for Bell-state/CHSH, MPS/POVM formulas, and random-sample computation statistics.

Expected: backup slides are clearly marked as optional and do not crowd the ten-minute main line.

### Task 3: Build the PPTX

**Files:**
- Create: `outputs/defense_ppt/中性原子量子接口仿真_答辩PPT.pptx`
- Create: `outputs/defense_ppt/assets/*`

- [ ] **Step 1: Convert selected PDF figures to PNG assets**

Use `python -c` with PyMuPDF to convert selected PDF figures at presentation-safe resolution into `outputs/defense_ppt/assets/`.

Expected: each selected PDF figure has a corresponding PNG asset with a transparent or white background suitable for PPT insertion.

- [ ] **Step 2: Generate the PPTX**

Use `python -c` with python-pptx to create the PPTX. Use a restrained academic design: 16:9 layout, HEU-compatible title page, clean white/light background, dark text, one claim per slide, large figure area, small footer only when needed for data-source clarification.

Expected: PPTX contains 11 main slides plus optional backup slides, has speaker notes only if python-pptx can add them reliably; otherwise write notes in `slide_outline.md`.

- [ ] **Step 3: Reopen the PPTX**

Run:

```powershell
python -c "from pptx import Presentation; p='outputs/defense_ppt/中性原子量子接口仿真_答辩PPT.pptx'; prs=Presentation(p); print(len(prs.slides))"
```

Expected: PPTX opens successfully and reports the expected slide count.

### Task 4: Render or Inspect Slides and Fix Layout

**Files:**
- Create: `outputs/defense_ppt/preview/*.png`
- Create: `outputs/defense_ppt/qa_report.md`
- Modify: `outputs/defense_ppt/中性原子量子接口仿真_答辩PPT.pptx`

- [ ] **Step 1: Try to locate a renderer**

Check whether LibreOffice `soffice`, PowerPoint-compatible conversion, or another already available headless renderer exists.

Expected: if a reliable renderer exists, export slides or PDF pages to PNG. If not, document the limitation and use package/object inspection plus available asset previews.

- [ ] **Step 2: Inspect rendered PNGs or slide objects**

Review slide previews for text overlap, unreadable figure labels, formula rendering problems, excessive bullet density, ambiguous data-source labels, and internal strategy wording.

Expected: every main slide has a clear title, a visible evidence object, and no obvious overlap.

- [ ] **Step 3: Fix the weakest slides**

Revise the PPTX generation command and rerun until main slides pass the visual checks. Prioritize slides 6, 7, 8, 9, and 10 because these carry the method, data-source boundary, and main results.

Expected: QA report records fixes and remaining limitations.

### Task 5: Final Review and Commit Planning Artifacts

**Files:**
- Modify: `docs/superpowers/specs/2026-05-26-thesis-defense-ppt-design.md`
- Add: `docs/superpowers/plans/2026-05-26-thesis-defense-ppt-plan.md`
- Keep local ignored outputs: `outputs/defense_ppt/*`

- [ ] **Step 1: Check git status**

Run:

```powershell
git status --short
```

Expected: tracked changes are limited to planning/design documents unless the user explicitly asks to track PPT outputs.

- [ ] **Step 2: Commit the design and plan update**

Commit only the tracked design/plan documents. Generated PPT outputs remain local unless explicitly requested.

Expected: repository has a clean tracked state and local PPT deliverables remain in `outputs/defense_ppt/`.

- [ ] **Step 3: Report final deliverables**

Report the PPTX path, preview/QA path, whether rendered PNG checks were available, and any remaining manual Morph-edit suggestions.

Expected: user can open the PPTX locally and refine Morph animations using their editable figure sources.
