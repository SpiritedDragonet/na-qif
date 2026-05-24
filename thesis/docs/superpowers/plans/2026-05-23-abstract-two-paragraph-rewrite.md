# Abstract Two-Paragraph Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the Chinese and English thesis abstracts into a clearer two-paragraph structure and reduce both keyword lists to five terms.

**Architecture:** The thesis stores cover metadata, abstracts, and keywords in `front/cover.tex`. The implementation is a focused text edit to that file, followed by local structural checks and a XeLaTeX validation pass.

**Tech Stack:** LaTeX, HeuThesis, PowerShell, `python -c`, `xelatex`.

---

### Task 1: Rewrite Abstracts and Keywords

**Files:**

Modify: `front/cover.tex:28-37`

Test: local checks against `front/cover.tex`, then `xelatex -interaction=batchmode -no-pdf thesis.tex`

- [ ] **Step 1: Re-read the approved design and current abstract**

Run: `Get-Content -Encoding UTF8 docs/superpowers/specs/2026-05-23-abstract-two-paragraph-rewrite-design.md`

Run: `$p='front/cover.tex'; $lines=Get-Content -Encoding UTF8 $p; for($i=28;$i -le 37;$i++){ Write-Output ("$i`t" + $lines[$i-1]) }`

Expected: the design calls for two synchronized abstract paragraphs, strong concrete results, and five keywords in each language.

- [ ] **Step 2: Replace the keyword lists**

Set `ckeywords` to `量子接口, 中性原子, 远程纠缠, 矩阵乘积态, 量子频率转换`.

Set `ekeywords` to `Quantum Interface, Neutral Atom, Remote Entanglement, Matrix Product State, Quantum Frequency Conversion`.

Expected: each keyword list contains exactly five entries.

- [ ] **Step 3: Rewrite the Chinese abstract**

Replace the single `cabstract` paragraph with two paragraphs. The first paragraph should explain the need for remote entanglement resources in quantum networks and distributed quantum computing, the coupled non-ideal factors in neutral-atom photon--atom interface links, and the need for traceable link simulation and error budgeting.

The second paragraph should state the concrete work and results. It should include time-bin discretization, MPS, TEBD, quantum trajectory sampling, Kraus projection, reconstruction of the atomic reduced density matrix, success and fidelity metrics, the 70 ns data window, the approximately 0.75 rad beam-splitter operating point, the 58.7% and 41.3% false-success decomposition, the higher true-success concentration at small time-bin differences, the distance-scaling conclusion, and the 99.42% detector-stage runtime bottleneck.

Expected: the Chinese abstract has exactly two paragraphs inside `\begin{cabstract}` and `\end{cabstract}`.

- [ ] **Step 4: Rewrite the English abstract**

Replace the single `eabstract` paragraph with two paragraphs that correspond to the Chinese abstract. The English version should use natural academic English and preserve the same technical objects, result sequence, and numerical conclusions.

Expected: the English abstract has exactly two paragraphs inside `\begin{eabstract}` and `\end{eabstract}`.

- [ ] **Step 5: Run structural checks**

Run: `python -c "from pathlib import Path; import re; t=Path('front/cover.tex').read_text(encoding='utf-8'); print('ckeywords', len([x for x in re.search(r'ckeywords=\\{([^}]*)\\}', t).group(1).split(',') if x.strip()])); print('ekeywords', len([x for x in re.search(r'ekeywords=\\{([^}]*)\\}', t).group(1).split(',') if x.strip()])); print('cabstract paragraphs', len([p for p in re.search(r'\\\\begin\\{cabstract\\}(.*?)\\\\end\\{cabstract\\}', t, re.S).group(1).split('\\n\\n') if p.strip()])); print('eabstract paragraphs', len([p for p in re.search(r'\\\\begin\\{eabstract\\}(.*?)\\\\end\\{eabstract\\}', t, re.S).group(1).split('\\n\\n') if p.strip()]))"`

Expected: `ckeywords 5`, `ekeywords 5`, `cabstract paragraphs 2`, `eabstract paragraphs 2`.

- [ ] **Step 6: Run LaTeX validation**

Run: `xelatex -interaction=batchmode -no-pdf thesis.tex`

Expected: exit code 0. If the log reports citation rerun warnings caused by earlier interrupted PDF writes, run `bibtex thesis`, then run `xelatex -interaction=batchmode -no-pdf thesis.tex` twice.

- [ ] **Step 7: Inspect the changed text**

Run: `git diff -- front/cover.tex`

Expected: only the keyword lists and abstract bodies changed in `front/cover.tex`. The text follows the approved two-paragraph logic and does not introduce unrelated formatting changes.

- [ ] **Step 8: Commit the abstract rewrite**

Run: `git add front/cover.tex docs/superpowers/plans/2026-05-23-abstract-two-paragraph-rewrite.md`

Run: `git commit -m "Revise thesis abstracts" -m "Co-Authored-By: Codex <noreply@openai.com>"`

Expected: the commit contains the plan and the focused abstract rewrite. Other pre-existing thesis changes remain unstaged unless the user explicitly asks to include them.
