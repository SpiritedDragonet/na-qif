# Thesis Figure System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first three chapters' figure-system revision described by `docs/figure_specs/visual_system_spec.md` and the three chapter-level figure specs.

**Architecture:** The implementation keeps reusable figure generation in `figures/`, updates chapter text in `body/chapter1.tex`, `body/chapter2.tex`, and `body/chapter3.tex`, and cleans obsolete figure paths after LaTeX references have moved. New self-drawn figures are generated as PDFs through reusable Python scripts; the user-supplied `fig_sim_pipeline.pdf` remains a static final artwork unless the user later provides a source file.

**Tech Stack:** LaTeX, XeLaTeX, Python, Matplotlib, existing `figures/plot_style.py`, `pdftoppm`, `view_image`.

---

### File Map

Create `figures/fig_ch1_scope_architecture.py` to generate `figures/fig_ch1_scope_architecture.pdf` and `figures/fig_ch1_scope_architecture.png`. This script owns the first-chapter architecture and research-scope figure that replaces `fig_tikz_interface_tasks` and `fig_tikz_arch_to_interface`.

Create `figures/fig_ch1_scheme_and_link.py` to generate `figures/fig_ch1_scheme_and_link.pdf` and `figures/fig_ch1_scheme_and_link.png`. This script owns the first-chapter scheme-selection and physical-link figure that replaces `fig_tikz_single_vs_two_photon` and absorbs the role of `Fig1.jpg`.

Create `figures/fig_ch2_timebin_mps_tebd.py` to generate `figures/fig_ch2_timebin_mps_tebd.pdf` and `figures/fig_ch2_timebin_mps_tebd.png`. This script owns the second-chapter three-panel method figure that replaces `PichlerZoller2015_Fig1_timebin.png`, `fig_mps_chain.pdf`, and `fig_tebd_pattern.jpg` as chapter figures.

Create `figures/fig_ch3_effect_pushback.py` to generate `figures/fig_ch3_effect_pushback.pdf` and `figures/fig_ch3_effect_pushback.png`. This script owns the third-chapter operator-flow version of the measurement-effect pushback figure.

Each newly generated self-drawn figure script saves the PDF used by LaTeX and a same-stem PNG used for immediate `view_image` inspection.

Modify `figures/run_all_figures.py` so it calls the new reusable scripts, removes obsolete script entries, and verifies the presence of static figures such as `fig_sim_pipeline.pdf` and `Zhou2023_Fig8_noise_vs_distance.png`.

Modify `body/chapter1.tex` to insert `fig_ch1_scope_architecture`, insert `fig_ch1_scheme_and_link`, keep `fig:noise_vs_distance`, and revise paragraphs around the merged figures.

Modify `body/chapter2.tex` to replace the three method figures with `fig_ch2_timebin_mps_tebd`, remove `fig:ch2-duallane`, and add the short cut-plane paragraph after `eq:ch2-op-impl-consistency`.

Modify `body/chapter3.tex` to remove `fig:ch3_effective_dofs`, add the table-to-formula bridge paragraph, replace the effect-pushback figure with `fig_ch3_effect_pushback`, and update the `fig:sim_pipeline` caption and surrounding text.

Delete obsolete generated/source files only after `rg` confirms that LaTeX and `run_all_figures.py` no longer reference them. Candidate cleanup includes `fig_tikz_interface_tasks.*`, `fig_tikz_arch_to_interface.*`, `fig_tikz_single_vs_two_photon.*`, `fig_tikz_ch2_duallane.*`, `fig_tikz_ch3_effective_dofs.*`, `fig_tikz_ch3_effect_pushback.*`, `fig_mps_chain.pdf`, and `fig_tebd_pattern.*`.

### Task 1: Baseline And Reference Inventory

**Files:**
- Read: `docs/figure_specs/visual_system_spec.md`
- Read: `docs/figure_specs/chapter1_figure_specs.md`
- Read: `docs/figure_specs/chapter2_figure_specs.md`
- Read: `docs/figure_specs/chapter3_figure_specs.md`
- Read: `body/chapter1.tex`
- Read: `body/chapter2.tex`
- Read: `body/chapter3.tex`
- Read: `figures/run_all_figures.py`

- [ ] **Step 1: Capture current working tree state**

Run: `git status --short`

Expected: Existing unrelated/generated changes are visible. Do not revert `figures/fig_sim_pipeline.pdf`, `figures/fig_sim_pipeline.py`, `thesis.pdf`, `thesis.ist`, `.codex_tmp`, or ignored agent-file work unless the user explicitly requests it.

- [ ] **Step 2: Confirm current figure references**

Run: `rg -n "fig:interface_tasks|fig:arch_to_interface|fig:single_vs_two_photon|fig:link_overview|fig:noise_vs_distance|fig:ch2-timebin|fig:ch2-mpschain|fig:ch2-tebd|fig:ch2-duallane|fig:ch3_effective_dofs|fig:ch3_effect_pushback|fig:sim_pipeline" body figures docs/figure_specs`

Expected: References match the planned replacement set in the four spec files.

- [ ] **Step 3: Inventory current reference images**

Run: `Get-ChildItem figures -File | Select-Object Name,Length,LastWriteTime | Sort-Object Name`

Expected: Existing reference assets include `Fig1.jpg`, `PichlerZoller2015_Fig1_timebin.png`, `Wang2017_Fig4_MPS_update.jpg`, `vanLeent2022_Fig1_interface_arch.jpg`, and `Zhou2023_Fig8_noise_vs_distance.png`.

- [ ] **Step 4: Render any candidate PDF reference that needs visual inspection**

Run examples:

```powershell
pdftoppm -png -singlefile figures\fig_tikz_ch2_duallane.pdf .codex_tmp\fig_tikz_ch2_duallane
pdftoppm -png -singlefile figures\fig_tikz_ch3_effect_pushback.pdf .codex_tmp\fig_tikz_ch3_effect_pushback
```

Expected: PNG files appear in `.codex_tmp`. Inspect them with `view_image` before using them as negative or positive references.

### Task 2: Chapter 1 Architecture Scope Figure

**Files:**
- Create: `figures/fig_ch1_scope_architecture.py`
- Create: `figures/fig_ch1_scope_architecture.pdf`
- Modify: `body/chapter1.tex`
- Modify: `figures/run_all_figures.py`
- Later cleanup: `figures/fig_tikz_interface_tasks.*`, `figures/fig_tikz_arch_to_interface.*`

- [ ] **Step 1: Draft the reusable figure script**

Create `figures/fig_ch1_scope_architecture.py` using Matplotlib. Reuse the local font setup pattern from existing figure scripts and import `figures/plot_style.py` if it provides usable defaults. The script should draw three horizontal regions: “可扩展性压力”, “模块化/分布式体系结构”, and “量子接口任务边界”. It should show two neutral-atom processing nodes with local registers, a communication atom or cavity interface, an orange photonic link, and a highlighted remote Bell-pair path.

- [ ] **Step 2: Generate the PDF**

Run: `python figures\fig_ch1_scope_architecture.py`

Expected: `figures/fig_ch1_scope_architecture.pdf` is created without Python exceptions.

- [ ] **Step 3: Render and inspect the PDF**

Run: `pdftoppm -png -singlefile figures\fig_ch1_scope_architecture.pdf .codex_tmp\fig_ch1_scope_architecture`

Expected: `figures\fig_ch1_scope_architecture.png` exists. Inspect it with `view_image` and verify that the three-region reading order, orange Bell-pair link, and research-scope highlight are clear.

- [ ] **Step 4: Update `body/chapter1.tex`**

Replace the early `fig:interface_tasks` figure and the later `fig:arch_to_interface` figure with one figure environment near the current `fig:arch_to_interface` location. Use the new include:

```tex
\includegraphics[width=0.96\textwidth]{fig_ch1_scope_architecture.pdf}
```

Keep a stable new label such as `fig:ch1_scope_architecture`. Update nearby prose so the first mention introduces the architecture pressure and research scope in paragraph form. Update downstream references that pointed to `fig:interface_tasks` or `fig:arch_to_interface`.

- [ ] **Step 5: Verify references**

Run: `rg -n "fig:interface_tasks|fig:arch_to_interface|fig_ch1_scope_architecture|fig:ch1_scope_architecture" body figures`

Expected: Old labels appear only in deleted or cleanup-candidate files. `body/chapter1.tex` uses the new label.

- [ ] **Step 6: Checkpoint**

Run: `git diff -- body/chapter1.tex figures/fig_ch1_scope_architecture.py figures/run_all_figures.py`

Expected: Diff contains only the architecture-scope figure and its chapter text. Commit only if the user asks for a checkpoint commit.

### Task 3: Chapter 1 Scheme And Link Figure

**Files:**
- Create: `figures/fig_ch1_scheme_and_link.py`
- Create: `figures/fig_ch1_scheme_and_link.pdf`
- Modify: `body/chapter1.tex`
- Modify: `figures/run_all_figures.py`
- Keep as reference or remove after confirmation: `figures/Fig1.jpg`
- Later cleanup: `figures/fig_tikz_single_vs_two_photon.*`

- [ ] **Step 1: Draft the figure script**

Create `figures/fig_ch1_scheme_and_link.py`. The figure should have two layers. The upper layer compares single-photon and double-photon routes with short Chinese labels for phase reference, single-click heralding, two-photon interference, mode matching, and coincidence success. The lower layer draws the adopted Rb atom-cavity, QFC, 1517 nm fiber, relay station, partial BSM, and detector chain. Use `Fig1.jpg` only as a structure and style reference.

- [ ] **Step 2: Generate and inspect**

Run: `python figures\fig_ch1_scheme_and_link.py`

Expected: `figures\fig_ch1_scheme_and_link.png` shows a readable two-layer figure. Inspect with `view_image`; verify that the top layer explains the route choice and the bottom layer defines the concrete link.

- [ ] **Step 3: Update `body/chapter1.tex`**

Replace the `fig:single_vs_two_photon` figure and `fig:link_overview` figure with one figure environment using:

```tex
\includegraphics[width=0.98\linewidth]{fig_ch1_scheme_and_link.pdf}
```

Use a stable label such as `fig:ch1_scheme_and_link`. Rewrite the paragraphs around the old two figures so the text first states route-selection criteria and then lands on the concrete link object.

- [ ] **Step 4: Preserve the literature evidence figure**

Keep `fig:noise_vs_distance` and its source `figures/Zhou2023_Fig8_noise_vs_distance.png`. Render or inspect the PNG with `view_image` and confirm the external-evidence role in the caption and surrounding prose.

- [ ] **Step 5: Verify references**

Run: `rg -n "fig:single_vs_two_photon|fig:link_overview|fig:ch1_scheme_and_link|Fig1.jpg|fig_tikz_single_vs_two_photon" body figures`

Expected: `body/chapter1.tex` references `fig:ch1_scheme_and_link`. Old labels appear only in cleanup-candidate files or historical aux files.

### Task 4: Chapter 2 Merged Method Figure

**Files:**
- Create: `figures/fig_ch2_timebin_mps_tebd.py`
- Create: `figures/fig_ch2_timebin_mps_tebd.pdf`
- Modify: `body/chapter2.tex`
- Modify: `figures/run_all_figures.py`
- Later cleanup: `figures/fig_mps_chain.pdf`, `figures/fig_tebd_pattern.*`, possibly unused direct chapter references to `PichlerZoller2015_Fig1_timebin.png`

- [ ] **Step 1: Draft the three-panel figure script**

Create `figures/fig_ch2_timebin_mps_tebd.py`. Panel `(a)` should map a continuous output field to time bins with the short formula `Delta B_k propto int_{t_k}^{t_{k+1}} b(t)dt`. Panel `(b)` should show A/B two-row MPS station layout with `emitter d=12`, repeated `bin d=5`, `memory d=3`, and one active window. Panel `(c)` should show one local TEBD update window with `U_emit`, `U_QFC`, `U_filt`, `SWAP`, and `SVD`.

- [ ] **Step 2: Generate and inspect**

Run: `python figures\fig_ch2_timebin_mps_tebd.py`

Expected: `figures\fig_ch2_timebin_mps_tebd.png` shows three panels with clear labels and no crowded local-basis text. Inspect with `view_image`.

- [ ] **Step 3: Update `body/chapter2.tex`**

Replace the separate `fig:ch2-timebin`, `fig:ch2-mpschain`, and `fig:ch2-tebd` figures with one figure environment using:

```tex
\includegraphics[width=0.98\textwidth]{fig_ch2_timebin_mps_tebd.pdf}
```

Use a stable label such as `fig:ch2_timebin_mps_tebd`. Adjust prose so the figure appears after the time-bin definition and before or around the chain mapping discussion. The text should connect panel `(a)` to time-bin operators, panel `(b)` to the Hilbert-space chain, and panel `(c)` to TEBD scheduling.

- [ ] **Step 4: Verify old method references**

Run: `rg -n "fig:ch2-timebin|fig:ch2-mpschain|fig:ch2-tebd|fig:ch2_timebin_mps_tebd|fig_mps_chain|fig_tebd_pattern|PichlerZoller2015_Fig1_timebin" body figures`

Expected: `body/chapter2.tex` references only the merged figure. External reference images may remain as source assets if they are useful for documentation, but chapter figures should use the merged self-drawn PDF.

### Task 5: Chapter 2 Dual-Lane Deletion And Text Bridge

**Files:**
- Modify: `body/chapter2.tex`
- Modify: `figures/run_all_figures.py`
- Later cleanup: `figures/fig_tikz_ch2_duallane.*`

- [ ] **Step 1: Remove the `fig:ch2-duallane` figure environment**

Edit `body/chapter2.tex` to remove the figure environment that includes `fig_tikz_ch2_duallane`.

- [ ] **Step 2: Add the cut-plane paragraph**

After `eq:ch2-op-impl-consistency` and before the success-event definitions, add one paragraph defining `rho_in`, `Phi`, `F_m`, and `E_m^{in}=Phi^\dagger(F_m)`. The paragraph should state that the input plane is a computational cut plane, `rho_in` is the state-side output at that plane, `Phi` is the downstream measurement-side channel, and the pushed-back effect is inserted into MPS contraction.

- [ ] **Step 3: Verify prose flow**

Run: `rg -n "rho_in|Phi\\^\\dagger|fig:ch2-duallane|fig_tikz_ch2_duallane|eq:ch2-op-impl-consistency" body/chapter2.tex`

Expected: The new paragraph appears near `eq:ch2-op-impl-consistency`, and no chapter text references `fig:ch2-duallane`.

### Task 6: Chapter 3 Model Boundary Cleanup

**Files:**
- Modify: `body/chapter3.tex`
- Modify: `figures/run_all_figures.py`
- Later cleanup: `figures/fig_tikz_ch3_effective_dofs.*`

- [ ] **Step 1: Remove `fig:ch3_effective_dofs`**

Delete the figure environment that includes `fig_tikz_ch3_effective_dofs`.

- [ ] **Step 2: Add the table-to-formula bridge paragraph**

After `tab:ch3_local_spaces`, add a short paragraph explaining that the table gives the model-space overview, then later sections combine atom and cavity into a 12D emitter, embed 780/1517 nm modes into a 5D time bin, represent filtering as a 5D bin with a 3D memory gate, and leave fiber/BS/detectors to measurement-side effects.

- [ ] **Step 3: Verify references**

Run: `rg -n "fig:ch3_effective_dofs|fig_tikz_ch3_effective_dofs|tab:ch3_local_spaces|12 维发射体|5 维时间" body/chapter3.tex figures`

Expected: Chapter text contains no reference to `fig:ch3_effective_dofs`; the table bridge paragraph exists.

### Task 7: Chapter 3 Effect-Pushback Figure

**Files:**
- Create: `figures/fig_ch3_effect_pushback.py`
- Create: `figures/fig_ch3_effect_pushback.pdf`
- Modify: `body/chapter3.tex`
- Modify: `figures/run_all_figures.py`
- Later cleanup: `figures/fig_tikz_ch3_effect_pushback.*`

- [ ] **Step 1: Draft the operator-flow figure script**

Create `figures/fig_ch3_effect_pushback.py`. The figure should draw the flow `F_m -> M_det^\dagger(F_m) -> Ad_{U_BS}^\dagger(...) -> L^\dagger(...) -> V^\dagger(...)V -> E_m^{in}`. Each arrow should be labeled as a mapping, and small physical labels should appear under the mappings: “探测器响应”, “分束器”, “光纤损耗/偏振”, and “模式嵌入回投影”. Add a final probability box `p_m=Tr(E_m^{in} rho_in)` with a short blue `rho_in` branch.

- [ ] **Step 2: Generate and inspect**

Run: `python figures\fig_ch3_effect_pushback.py`

Expected: `figures\fig_ch3_effect_pushback.png` reads as a mathematical operator-flow diagram. Inspect with `view_image`; verify that it reads as a mathematical pushback flow.

- [ ] **Step 3: Update `body/chapter3.tex`**

Move the figure to the paragraph after `E_m^{in}=Phi^\dagger(F_m)` and `Phi=M_det circ Ad_{U_BS} circ L circ V`. Include:

```tex
\includegraphics[width=0.96\linewidth]{fig_ch3_effect_pushback.pdf}
```

Keep the label `fig:ch3_effect_pushback` unless a label rename is required. Update the caption to emphasize the order relation between `Phi` and `Phi^\dagger`.

- [ ] **Step 4: Verify references**

Run: `rg -n "fig:ch3_effect_pushback|fig_tikz_ch3_effect_pushback|fig_ch3_effect_pushback|Phi\\^\\dagger|E_m\\^\\{in\\}" body/chapter3.tex figures/run_all_figures.py`

Expected: `body/chapter3.tex` includes the new PDF and the label remains stable.

### Task 8: Chapter 3 Simulation Pipeline Integration

**Files:**
- Modify: `body/chapter3.tex`
- Modify: `figures/run_all_figures.py`
- Static input: `figures/fig_sim_pipeline.pdf`

- [ ] **Step 1: Wait for or confirm the user-supplied final artwork**

Confirm that `figures/fig_sim_pipeline.pdf` contains the finalized Chinese labels and no red placeholder text. If the user supplies a new PDF or image, place it at `figures/fig_sim_pipeline.pdf`.

- [ ] **Step 2: Render and inspect**

Run: `pdftoppm -png -singlefile figures\fig_sim_pipeline.pdf .codex_tmp\fig_sim_pipeline_final`

Expected: `.codex_tmp\fig_sim_pipeline_final.png` exists. Inspect with `view_image`; verify that the second row and bottom row carry the main data-flow story and the right side reads as a metrics entrance.

- [ ] **Step 3: Update caption and surrounding prose**

In `body/chapter3.tex`, revise the caption and following paragraph so they match the simplified spec: state-side MPS generates `rho_in`, measurement-side effect construction generates `E_d^{in}`, and the right-side formula performs conditional contraction for the fourth-chapter metrics.

- [ ] **Step 4: Update `run_all_figures.py` for static figures**

Modify `figures/run_all_figures.py` so `fig_sim_pipeline.pdf` is checked as an existing static figure rather than executed through a missing `fig_sim_pipeline.py`. Use a small list such as:

```python
static_figures = [
    "fig_sim_pipeline.pdf",
    "Zhou2023_Fig8_noise_vs_distance.png",
]
```

For each static figure, raise `FileNotFoundError` if it is missing.

- [ ] **Step 5: Verify references**

Run: `rg -n "fig_sim_pipeline.py|fig_sim_pipeline.pdf|fig:sim_pipeline|E_d\\^\\{in\\}|rho_in" figures/run_all_figures.py body/chapter3.tex`

Expected: `fig_sim_pipeline.py` is not required by `run_all_figures.py`; `body/chapter3.tex` references `fig_sim_pipeline.pdf`.

### Task 9: Obsolete Figure Cleanup

**Files:**
- Delete after verification: obsolete source/render files listed in the File Map
- Modify if needed: `figures/run_all_figures.py`

- [ ] **Step 1: Confirm obsolete references are gone**

Run:

```powershell
rg -n "fig_tikz_interface_tasks|fig_tikz_arch_to_interface|fig_tikz_single_vs_two_photon|fig_tikz_ch2_duallane|fig_tikz_ch3_effective_dofs|fig_tikz_ch3_effect_pushback|fig_mps_chain|fig_tebd_pattern" body figures docs/figure_specs
```

Expected: Active `body/*.tex` and `figures/run_all_figures.py` contain no obsolete references. Spec files may still mention old names as historical context.

- [ ] **Step 2: Delete obsolete files**

Use `git rm` for tracked obsolete files and plain delete for untracked generated remnants. Keep external reference images that are still cited, useful for reference, or explicitly retained by the specs.

- [ ] **Step 3: Verify generation script inventory**

Run: `python figures\run_all_figures.py`

Expected: New and existing reusable scripts run successfully. Static required figures are present.

### Task 10: Full Thesis Verification

**Files:**
- Read generated logs: `thesis.log`
- Output: `thesis.pdf`

- [ ] **Step 1: Generate all scripted figures**

Run: `python figures\run_all_figures.py`

Expected: Exit code 0. All expected PDFs exist in `figures/`.

- [ ] **Step 2: Build the thesis**

Run these commands from the repo thesis directory:

```powershell
xelatex -interaction=nonstopmode thesis.tex
bibtex thesis
xelatex -interaction=nonstopmode thesis.tex
xelatex -interaction=nonstopmode thesis.tex
```

Expected: Exit code 0 for each command and an updated `thesis.pdf`.

- [ ] **Step 3: Scan for broken references**

Run: `rg -n "undefined|Undefined|Citation.*undefined|Reference.*undefined|File .* not found" thesis.log`

Expected: No matches related to the changed figures, labels, or citations.

- [ ] **Step 4: Render changed figure PDFs to PNG**

Run:

```powershell
pdftoppm -png -singlefile figures\fig_ch1_scope_architecture.pdf .codex_tmp\fig_ch1_scope_architecture_final
pdftoppm -png -singlefile figures\fig_ch1_scheme_and_link.pdf .codex_tmp\fig_ch1_scheme_and_link_final
pdftoppm -png -singlefile figures\fig_ch2_timebin_mps_tebd.pdf .codex_tmp\fig_ch2_timebin_mps_tebd_final
pdftoppm -png -singlefile figures\fig_ch3_effect_pushback.pdf .codex_tmp\fig_ch3_effect_pushback_final
pdftoppm -png -singlefile figures\fig_sim_pipeline.pdf .codex_tmp\fig_sim_pipeline_final
```

Expected: PNG files exist. Inspect each with `view_image`; confirm readability, arrow paths, color consistency, and absence of placeholder text.

- [ ] **Step 5: Review final diff**

Run: `git diff --stat`

Run: `git diff -- body/chapter1.tex body/chapter2.tex body/chapter3.tex figures/run_all_figures.py`

Expected: Diff matches the figure-system revision. Unrelated pre-existing changes remain untouched.

- [ ] **Step 6: Checkpoint**

Run: `git status --short`

Expected: New figure scripts, generated PDFs, chapter text updates, spec files, and cleanup deletions are visible. Create a commit only after the user explicitly requests it.
