# Conclusion Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the thesis conclusion into a short opening, three numbered conclusion paragraphs, and a restrained closing paragraph.

**Architecture:** This is a single-file thesis prose edit. The existing one-paragraph conclusion in `back/conclusion.tex` will be replaced with a result-focused conclusion that preserves the key numeric findings while avoiding a method recap or repeated self-credit.

**Tech Stack:** LaTeX thesis source, XeLaTeX verification, prose review against `AGENTS.md`.

---

### Task 1: Rewrite Conclusion Structure

**Files:**
- Modify: `back/conclusion.tex`

- [ ] **Step 1: Preserve source facts**

Read the current conclusion and retain these result facts unless the surrounding chapter contradicts them: `26.2 ns` excited-state lifetime scale, QFC efficiency `0.57`, `27 MHz` filter-cavity bandwidth, `70 ns` data window, `208 ns` hardware coincidence window, `160--170 s^{-1}` central-station background count rate, `330 us` long-link atomic coherence time, `0.955(7)` HOM visibility, recommended splitter point near `0.75 rad`, background-assisted false-success fraction about `58.7%`, intrinsic dark-count-assisted false-success fraction about `41.3%`, small time-bin difference cumulative true-event share about `47.6%`, and long-distance success probability/quality trends.

- [ ] **Step 2: Replace the one-paragraph conclusion**

Rewrite `back/conclusion.tex` so the body between `\begin{conclusions}` and `\end{conclusions}` has this shape: one short opening paragraph; three numbered paragraphs beginning with `第一，`, `第二，`, and `第三，`; one short closing paragraph.

- [ ] **Step 3: Keep conclusion logic result-first**

Ensure the first numbered paragraph concludes that link quality must be evaluated by true success, false success, and conditional-state quality together. Ensure the second numbered paragraph concludes that the usable operating point is jointly determined by window selection, HOM quality, and BSM record authenticity. Ensure the third numbered paragraph concludes that long-distance operation is first rate-limited and then increasingly noise-limited, producing a practical tuning order.

- [ ] **Step 4: Apply style constraints**

Keep each numbered item as a complete prose paragraph, not a bullet list with short fragments. Avoid second-level lists. Use `本文` only if needed in the opening. Do not repeat `本文完成了` or `本文提出了`. Use public experimental data only as baseline anchors or comparison points, not as if they were the conclusion itself.

- [ ] **Step 5: Reviewer-perspective self-check**

Read the rewritten conclusion as an internal-review teacher. Check whether any sentence sounds like a method recap, self-praise, or defensive ownership claim. Check whether a reviewer can quickly identify the three main conclusions and the practical significance of each one.

- [ ] **Step 6: Compile and inspect logs**

Run `xelatex -interaction=batchmode -no-pdf thesis.tex`. If LaTeX asks for rerun, run it once more. Inspect `thesis.log` with a search for `Undefined`, `Citation`, `Error`, and `Rerun`. Existing `pgfplots` compatibility and missing glossary warnings can remain if unchanged. Run `git diff -- back/conclusion.tex` before reporting to verify the thesis prose edit stayed scoped to the conclusion file.
