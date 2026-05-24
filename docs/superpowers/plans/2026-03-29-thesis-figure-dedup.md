# Thesis Figure Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收口第四章重复图文，调整复杂度图位置，并处理第一章与第三章之间的跨章重复叙述。

**Architecture:** 保留现有图号标签体系中的主干角色，但重新划分各图职责。第四章把 QFC 与探测端两张器件参数图合并为同一器件层工作平面，误差预算图继续承担系统级综合判断；第一章保留物理方案选择，第三章保留实现流程映射。

**Tech Stack:** LaTeX, Matplotlib, Python figure scripts

---

### Task 1: 合并第四章器件层图

**Files:**
- Modify: `thesis/figures/fig_qfc_noise.py`
- Delete: `thesis/figures/fig_detector_eff.py`
- Modify: `thesis/figures/run_all_figures.py`
- Modify: `thesis/body/chapter4.tex`

- [ ] 读取 QFC 与探测端两组 summary 数据
- [ ] 将 `fig_qfc_noise.py` 改为三联图，统一呈现 QFC 平面与探测平面
- [ ] 删除单独的 `fig_detector_eff.py` 入口，避免双路径并存
- [ ] 修改第四章对应图题、解释文字与前后衔接

### Task 2: 调整复杂度图在第四章中的落位

**Files:**
- Modify: `thesis/body/chapter4.tex`

- [ ] 在复杂度小节前后加入浮动体边界控制
- [ ] 确保复杂度图留在本节内部，不漂移到章节总结外

### Task 3: 处理第一章与第三章的跨章去重

**Files:**
- Modify: `thesis/body/chapter1.tex`
- Modify: `thesis/body/chapter3.tex`

- [ ] 第一章只保留方案选择与物理链路边界
- [ ] 第三章把流程图明确为“实现映射”，不重复绪论中的方案介绍
- [ ] 用交叉引用明确图 1.3、图 1.4 与图 3.3 的不同职责

### Task 4: 生成并验证

**Files:**
- Modify: `thesis/figures/fig_qfc_noise.pdf`
- Modify: `thesis/thesis.pdf`

- [ ] 重新生成修改后的图 PDF
- [ ] 运行 `xelatex` 两遍更新图号与交叉引用
- [ ] 核对新的 `aux` 编号与正文引用是否一致
