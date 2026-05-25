# Review Response Markdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `评阅意见修改建议汇总.md` into a formal, directly reusable thesis review-response document that quotes the original “论文的不足之处和建议” from `阅1.pdf` and `阅2.pdf` and explains the corresponding revisions in detail.

**Architecture:** The work is a single-document transformation. `阅1.pdf` maps to “评阅专家2” and `阅2.pdf` maps to “评阅专家3”; their original suggestions should be grouped by topic, then answered with the existing revision records rewritten into formal prose. The final file should keep enough structure for copying into the official modification statement while following `AGENTS.md` prose-first writing rules.

**Tech Stack:** Markdown, PowerShell, `pdftotext`, `rg`, manual `apply_patch` edits.

---

### Task 1: Reconfirm Inputs And Extract Expert Opinion Text

**Files:**
- Read: `阅1.pdf`
- Read: `阅2.pdf`
- Read: `评阅意见修改建议汇总.md`
- Read: `docs/superpowers/specs/2026-05-25-review-response-md-design.md`

- [ ] **Step 1: Re-read the design spec**

Run:

```powershell
Get-Content docs\superpowers\specs\2026-05-25-review-response-md-design.md
```

Expected: The spec states that `阅1.pdf` is “评阅专家2”, `阅2.pdf` is “评阅专家3”, and the final document must follow `AGENTS.md` writing rules.

- [ ] **Step 2: Extract `阅1.pdf` opinion text**

Run:

```powershell
pdftotext -layout "阅1.pdf" -
```

Expected: The output includes the “论文的不足之处和建议” section from 评阅专家2. Keep the original numbered items and the specific examples under item 2, including the abstract, page, formula, figure/table, and reference-format examples.

- [ ] **Step 3: Extract `阅2.pdf` opinion text**

Run:

```powershell
pdftotext -layout "阅2.pdf" -
```

Expected: The output includes the “论文的不足之处和建议” section from 评阅专家3. Keep items 1 through 4 as substantive revision suggestions. Item 5 is encouragement and should only be mentioned, if needed, under a brief non-modification note.

- [ ] **Step 4: Re-read the existing md**

Run:

```powershell
$p='评阅意见修改建议汇总.md'; $lines=Get-Content -LiteralPath $p; for($i=1;$i -le $lines.Length;$i++){ Write-Output ("$i`t" + $lines[$i-1]) }
```

Expected: The current checklist content is visible, including all existing “修改记录” paragraphs that will be reused and rewritten.

### Task 2: Build The Formal Document Structure

**Files:**
- Modify: `评阅意见修改建议汇总.md`

- [ ] **Step 1: Replace the internal checklist title**

Change the title from internal summary wording to formal response wording, for example:

```markdown
# 学位论文评阅意见修改说明
```

Add a short opening paragraph explaining that the document responds to the “论文的不足之处和建议” from 评阅专家2 and 评阅专家3, and that the revision records are organized by topic for clarity.

- [ ] **Step 2: Remove checkbox syntax**

Remove all `- [x]` checklist markers. Each former checklist item should become either a `###` subsection or be merged into a broader `###` subsection when it responds to the same expert opinion.

Run after editing:

```powershell
rg -n "\[[ xX]\]" "评阅意见修改建议汇总.md"
```

Expected: No matches.

- [ ] **Step 3: Use the standard response block**

Each substantive response should use this structure:

```markdown
### N. 关于……的问题

评阅专家意见：
评阅专家2指出：“……”
评阅专家3指出：“……”

修改情况：
……
```

If only one expert raised the issue, list only that expert. If an item is based on internal review or later self-check rather than the two external expert PDFs, use wording such as “根据内审意见及后续全文自查” and do not label it as an external expert opinion.

### Task 3: Quote Expert Opinions Under The Correct Topics

**Files:**
- Modify: `评阅意见修改建议汇总.md`

- [ ] **Step 1: Add 评阅专家2 original opinions**

Use the original text from `阅1.pdf` for these substantive items:

```text
1、建议在第一章中添加该方向的国内外研究现状。
2、论文存在语言艰涩难懂，描述不准确或不清楚问题，建议认真检查论文语言表述并加以修改。主要问题如……
3、文中出现“partial BSM”和“部分 BSM”、“时间分箱”和“时间bin”等不同表述……
4、文中没有给出“CHSH”，“HOM”的英文全拼……
5、文中表2.1，表2.2，表3.1，图3.1在文中没有说明也没有引用……
6、第48页倒数第3行……
7、第49页第2段讲述图4.8与图4.10(b)……
8、论文存在大量参考文献引用格式不正确……
```

Expected: The specific examples under item 2 and item 8 are preserved, not compressed into a generic summary.

- [ ] **Step 2: Add 评阅专家3 original opinions**

Use the original text from `阅2.pdf` for these substantive items:

```text
1.文中多处出现分号使用不当的情况……
2.P20公式2-27表述不规范……
3.表3.2与图3.6被置于章节的小结部分……
4.文中多个位置出现不必要的空白页……
```

Expected: These items are placed under matching topics rather than isolated at the end if a current topic already responds to them.

- [ ] **Step 3: Combine overlapping issues**

Combine related expert opinions into one response when appropriate. At minimum:

The language-readability section should include 评阅专家2 item 2 and may also reference 评阅专家3 item 1 when discussing punctuation and readability, while keeping semicolon handling as a dedicated subtopic if clearer.

The figure/table section should include 评阅专家2 item 5 and 评阅专家3 item 3.

The formatting and layout section should include 评阅专家3 item 4 and any existing layout checks.

The reference-format section should include 评阅专家2 item 8 and the existing reference audit details.

### Task 4: Rewrite Modification Records As Formal Prose

**Files:**
- Modify: `评阅意见修改建议汇总.md`

- [ ] **Step 1: Convert existing “修改记录” paragraphs**

Rewrite each “修改记录：” paragraph into a formal “修改情况：” paragraph. Keep the concrete details already present in the old md: chapter names, exact sections, terminology changes, figure/table movements, reference-format checks, and compilation or review outcomes.

Expected: The final wording reads like a submission-ready response to reviewers, not an internal work log.

- [ ] **Step 2: Preserve internal-review and self-check items separately**

For items that do not come directly from the two PDFs, keep them only when they materially support the formal response. Examples include the abstract two-paragraph rewrite, conclusion rewrite, reference authenticity audit, `/OL` handling, Japanese source replacement, and “错误综合征” terminology clarification.

Expected: These items are introduced as “根据内审意见及后续全文自查” or similar, never as “评阅专家2/3指出”.

- [ ] **Step 3: Apply `AGENTS.md` writing rules**

Use coherent paragraphs. Avoid turning each response into many short bullet points. Use lists only for expert original numbered items or when several independent expert comments must be preserved exactly.

Expected: The prose is readable, formal, and directly reusable in the official modification statement.

### Task 5: Verify Coverage And Formatting

**Files:**
- Verify: `评阅意见修改建议汇总.md`

- [ ] **Step 1: Check that checkbox syntax is gone**

Run:

```powershell
rg -n "\[[ xX]\]" "评阅意见修改建议汇总.md"
```

Expected: No matches.

- [ ] **Step 2: Check that internal source labels are gone**

Run:

```powershell
rg -n "来源：|阅1|阅2|第3页|第4页" "评阅意见修改建议汇总.md"
```

Expected: No matches for `来源：`, `阅1`, or `阅2`. Page references may remain only if they are part of the quoted expert opinion itself, but the preferred final wording should rely on “评阅专家2/3指出”.

- [ ] **Step 3: Check expert labels**

Run:

```powershell
rg -n "评阅专家[23]指出" "评阅意见修改建议汇总.md"
```

Expected: Multiple matches exist, and every substantive external expert opinion is attributed to either 评阅专家2 or 评阅专家3.

- [ ] **Step 4: Check for required quoted keywords**

Run:

```powershell
rg -n "国内外研究现状|语言艰涩|partial BSM|时间bin|CHSH|HOM|表2\\.1|公式2-27|表3\\.2|图3\\.6|空白页|参考文献引用格式" "评阅意见修改建议汇总.md"
```

Expected: All major expert-suggestion anchors are present.

- [ ] **Step 5: Review the whole file manually**

Run:

```powershell
$p='评阅意见修改建议汇总.md'; $lines=Get-Content -LiteralPath $p; for($i=1;$i -le $lines.Length;$i++){ Write-Output ("$i`t" + $lines[$i-1]) }
```

Expected: The document can be copied into the official modification statement without further structural editing.

### Task 6: Commit The Markdown Rewrite

**Files:**
- Commit: `评阅意见修改建议汇总.md`

- [ ] **Step 1: Review diff**

Run:

```powershell
git diff -- "评阅意见修改建议汇总.md"
```

Expected: The diff shows the checklist transformed into formal response text. No unrelated files are included in this task.

- [ ] **Step 2: Stage only the target md**

Run:

```powershell
git add -- "评阅意见修改建议汇总.md"
```

Expected: Only the target md is staged.

- [ ] **Step 3: Commit**

Run:

```powershell
git commit -m "docs: rewrite review response summary" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

Expected: A commit is created for the markdown rewrite. Do not include thesis source, generated PDFs, or other dirty files unless explicitly requested.
