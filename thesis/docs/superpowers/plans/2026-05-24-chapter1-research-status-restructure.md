# Chapter 1 Research Status Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure Chapter 1 so the domestic and international research status is visible, specific, and aligned with reviewer expectations while preserving the detailed neutral-atom interface and link-bottleneck content from baseline commit `6a71ea45`.

**Architecture:** This is a single-chapter LaTeX restructure. The work should primarily move and relabel existing prose in `thesis/body/chapter1.tex`, with only local connective rewrites at section boundaries. The baseline commit `6a71ea45` is the source for detail-preservation checks; the implementation must prove that the current detailed content in old `1.3` and `1.4` did not collapse into a shorter generic summary.

**Tech Stack:** LaTeX, XeLaTeX, BibTeX, Git, PowerShell, ripgrep.

---

### File Structure

Only `thesis/body/chapter1.tex` should be modified during the main restructuring task. `thesis/thesis.aux`, `thesis/body/chapter1.aux`, `thesis/thesis.bbl`, `thesis/thesis.blg`, `thesis/thesis.out`, and `thesis/thesis.toc` may change during verification runs. Do not modify `thesis/reference.bib` unless LaTeX/BibTeX reports a real citation problem caused by the restructure. Do not create new one-off scripts; use direct PowerShell or `python -c` commands for temporary checks, following `AGENTS.md`.

The design document is `thesis/docs/superpowers/specs/2026-05-24-chapter1-research-status-restructure-design.md`. The baseline commit for comparison is `6a71ea45 Save thesis review revision baseline`. The final implementation should be committed separately after verification.

### Task 1: Baseline Orientation

**Files:**
Modify: none.

- [ ] **Step 1: Confirm current branch and clean worktree**

Run from repository root `G:\BProj\Quantum_simulation`:

```powershell
git branch --show-current
git status --short
```

Expected: branch is `main`; status is empty before editing.

- [ ] **Step 2: Re-read the design spec**

Run:

```powershell
Get-Content thesis\docs\superpowers\specs\2026-05-24-chapter1-research-status-restructure-design.md
```

Expected: the plan follows the approved structure and the 18-item detail-preservation checklist.

- [ ] **Step 3: Inspect current chapter headings**

Run:

```powershell
rg -n "^\\section|^\\subsection" thesis\body\chapter1.tex
```

Expected: old headings include `量子计算简介`, `国内外研究现状`, `中性原子量子接口链路`, `链路级性能瓶颈分析`, and `主要研究内容与结构安排`.

### Task 2: Restructure Headings and Section Boundaries

**Files:**
Modify: `thesis/body/chapter1.tex`.

- [ ] **Step 1: Rename the background section and subsections**

In `thesis/body/chapter1.tex`, rename:

```tex
\section{量子计算简介}[Introduction to quantum computing]
\subsection{量子计算基本概念与模型}
\subsection{工程化价值与应用牵引}
\subsection{可扩展性压力与模块化思路}
```

to:

```tex
\section{量子计算研究背景}[Research background of quantum computing]
\subsection{量子计算基本概念与模型}
\subsection{量子计算的工程化应用}
\subsection{可扩展量子计算体系结构需求}
```

Keep the existing paragraphs, figures, and tables in this section. Adjust only local transition phrases if the old wording refers to the old section title.

- [ ] **Step 2: Expand `国内外研究现状` into concrete subsections**

Replace the current single-paragraph `\section{国内外研究现状}` area with this heading structure:

```tex
\section{国内外研究现状}[Domestic and international research status]
\subsection{可扩展量子计算体系结构研究进展}
\subsection{量子接口平台研究进展}
\subsection{中性原子处理节点研究进展}
\subsection{长距离原子--光子纠缠链路研究进展}
\subsection{链路性能建模研究进展}
```

Do not write generic filler. Each subsection must inherit concrete material from the current Chapter 1. The first two subsections may use the current `1.2` paragraph plus surrounding background content as connecting prose. The latter three subsections must migrate the substance of current `1.3` and `1.4`.

- [ ] **Step 3: Rename the final research-content section**

Rename:

```tex
\section{主要研究内容与结构安排}[Main contributions and organization]
\subsection{链路建模方法概述}
\subsection{研究内容与章节安排}
```

to:

```tex
\section{本文研究内容}[Research contents of this thesis]
\subsection{研究对象}
\subsection{建模方法}
\subsection{章节安排}
```

Move the first part of the current `研究内容与章节安排` paragraph into `研究对象`, keep the MPS/TEBD/POVM material in `建模方法`, and leave the final chapter-order sentences in `章节安排`.

### Task 3: Migrate the Current `1.3` Technical Detail

**Files:**
Modify: `thesis/body/chapter1.tex`.

- [ ] **Step 1: Move neutral-atom node framing into `中性原子处理节点研究进展`**

Move or adapt the current prose beginning with `中性原子节点是指一类以中性原子为载体` into `\subsection{中性原子处理节点研究进展}`. Preserve the distinction between communication atoms/modes and computation atoms, and preserve the connection to processing neutral-atom nodes and modular/distributed quantum computing.

- [ ] **Step 2: Move the representative link route into `长距离原子--光子纠缠链路研究进展`**

Move or adapt the current prose describing `$^{87}$Rb + 高精细度光学腔 + QFC + 电信光纤 + 中继站 BSM`. Preserve details about `$^{87}$Rb`, cavity enhancement, QFC, the telecom band, `$1517\,\mathrm{nm}$`, two arms of about `$16\,\mathrm{km}$`, classical communication delay, two-photon interference, and partial BSM.

- [ ] **Step 3: Keep `fig_link_overview` near the representative-link discussion**

Keep the existing `fig_link_overview` figure close to the subsection where the representative chain is described. Preserve the caption details for Node A/B, QFC, `L_1,L_2\le 16.5\,\mathrm{km}`, BS, PBS, HWP/QWP, PPLN, SNSPD, and the existing citations.

- [ ] **Step 4: Preserve link-selection reasoning**

Keep the current logic that the selected interface link matches a modular/distributed quantum-computing setting where local neutral-atom arrays handle processing and communication qubits establish remote Bell pairs. Keep the distinction from quantum-repeaters whose priority is storage lifetime, multimode multiplexing, and network-layer swapping.

### Task 4: Migrate the Current `1.4` Bottleneck Detail

**Files:**
Modify: `thesis/body/chapter1.tex`.

- [ ] **Step 1: Move success-rate bottleneck prose into `链路性能建模研究进展`**

Move or adapt the current `成功率与距离代价` subsection. Preserve the multiplicative collapse of heralding probability, the factors of emission, collection/coupling, QFC, fiber transmission, BSM, and detection efficiency, and the effect of distance on fiber loss and announcement waiting time.

- [ ] **Step 2: Move fidelity and noise-limit prose into `链路性能建模研究进展`**

Move or adapt the current `保真度退化与噪声受限机制` subsection. Preserve indistinguishability, frequency/time/polarization/spatial-mode mismatch, dark counts, QFC background, scattering leakage, false heralding, efficiency-limited to noise-limited transition, filtering memory, detector gate width, and coincidence windows.

- [ ] **Step 3: Keep the noise-distance figure**

Keep the existing `fig:noise_vs_distance` figure near the discussion of efficiency-limited and noise-limited regimes. Preserve the citation to `zhou2024longlived`.

- [ ] **Step 4: Add a concise research-gap transition**

At the end of `链路性能建模研究进展`, add a short paragraph in prose explaining that these bottlenecks require a unified link model that handles continuous-time propagation, conditional measurement, temporal windows, and multiple noise sources. This paragraph should naturally lead into `本文研究内容`.

### Task 5: Rewrite `本文研究内容` Without Losing Method Detail

**Files:**
Modify: `thesis/body/chapter1.tex`.

- [ ] **Step 1: Write `研究对象` from existing prose**

Use the current first half of `研究内容与章节安排` to state the object of this thesis: a heralded remote-entanglement link consisting of atom-photon emission, QFC, telecom-fiber propagation, two-photon interference, and partial BSM. Preserve the metrics of heralding statistics, remote-entanglement generation rate, conditional-state quality, dominant error sources, and parameter scans.

- [ ] **Step 2: Write `建模方法` from existing prose**

Use the current `链路建模方法概述` material. Preserve time-bin discretization, MPS, TEBD, continuous-time open-system evolution, propagation delay, conditional measurement, POVM, and the idea of pushing measurement effects back to the node state.

- [ ] **Step 3: Write `章节安排` from existing prose**

Keep the existing chapter-order information. Use prose, not a list, following `AGENTS.md`.

### Task 6: Detail-Preservation Check Against Baseline

**Files:**
Modify: `thesis/body/chapter1.tex` only if this task finds missing details.

- [ ] **Step 1: Generate a baseline comparison diff**

Run:

```powershell
git diff 6a71ea45 -- thesis/body/chapter1.tex
```

Expected: changes are mostly heading changes, movement, and local connective rewrites. Large blocks of detailed old `1.3` and `1.4` content should appear as moved or adapted material, not disappeared without replacement.

- [ ] **Step 2: Check the 18 detail-preservation items**

Use direct search commands such as:

```powershell
rg -n "通信比特|计算比特|87|Rb|高精细度|1517|16|BSM|暗计数|假宣告|噪声受限|时间窗|MPS|TEBD|POVM" thesis\body\chapter1.tex
```

Expected: each of the 18 checklist items from the spec is findable in the revised Chapter 1. If a detail is missing or reduced to a vague phrase, restore it from `6a71ea45`.

- [ ] **Step 3: Check title specificity**

Run:

```powershell
rg -n "^\\section|^\\subsection" thesis\body\chapter1.tex
```

Expected: no vague placeholder title such as `现有研究存在的问题`; no repeated decorative `xxx与xxx` title pattern; each research-status subsection names a concrete technical object.

### Task 7: LaTeX Verification

**Files:**
Generated/modified by tools: `thesis/thesis.aux`, `thesis/body/chapter1.aux`, `thesis/thesis.out`, `thesis/thesis.toc`, and possibly `thesis/thesis.bbl`, `thesis/thesis.blg`.

- [ ] **Step 1: Run XeLaTeX without writing PDF**

Run from `G:\BProj\Quantum_simulation\thesis`:

```powershell
xelatex -interaction=batchmode -no-pdf thesis.tex
```

Expected: exit code 0.

- [ ] **Step 2: Run BibTeX only if citations changed or warnings require it**

If XeLaTeX reports undefined citations or the `.aux` changed citation order in a way that requires refreshing the bibliography, run:

```powershell
bibtex thesis
xelatex -interaction=batchmode -no-pdf thesis.tex
xelatex -interaction=batchmode -no-pdf thesis.tex
```

Expected: BibTeX warning count is 0; XeLaTeX exits 0.

- [ ] **Step 3: Search logs for citation and label problems**

Run:

```powershell
Select-String -Path thesis.log,thesis.blg -Pattern "undefined|Undefined|Citation|Label|multiply defined|not found|Error|There were"
```

Expected: no undefined citation, missing figure, or duplicate-label problem introduced by the restructure. MiKTeX log-write permission messages can be ignored if the TeX command exits 0.

### Task 8: Final Review and Commit

**Files:**
Modify: `thesis/body/chapter1.tex` and generated LaTeX outputs from verification.

- [ ] **Step 1: Review final diff**

Run:

```powershell
git diff --stat
git diff -- thesis/body/chapter1.tex
```

Expected: the chapter diff clearly shows structure improvement and local prose changes. It should not show wholesale deletion of detailed chain descriptions without corresponding relocated content.

- [ ] **Step 2: Commit the implementation**

Run from repository root:

```powershell
git add -A
git commit -m "Revise chapter 1 research status structure" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: commit succeeds. Do not push unless explicitly requested.

- [ ] **Step 3: Report the outcome**

Report the implementation commit hash, the verification commands and outcomes, and a short mapping from old sections to new sections. Include whether any detail from the preservation checklist required restoration after diff review.
