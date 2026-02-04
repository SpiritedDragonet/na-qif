# 5D 大改（QFC/光纤/BS 推入 POVM）修改计划

> 目标：取消 18D bin，改为 5D bin；发射之后的 **QFC + 780 过滤 + 光纤噪声/损耗 + BS** 全部缩进 POVM（Heisenberg/对偶映射）。  
> 约束：不保留旧方案残留；注释量保持现有风格；不保留 `apply_qfc/apply_bs` 占位接口。

---

## 0) 已阅读范围（确保计划“打通逻辑”所需的核心文件）

- `README.md`（项目结构与流程描述）
- `atom_sim/hilbert/basis.py`, `operators.py`
- `atom_sim/physics/gates.py`, `channels.py`
- `atom_sim/simulation/trajectory.py`, `detection.py`
- `atom_sim/visualization/wavepacket.py`
- `atom_sim/experiment/single_run.py`, `hom.py`
- `total_simulation.py`
- `docs/28(已完成)_专家对于当前程序的修改意见3.md`
- `docs/30(已过时)_有关更前面的780nmQFC等门推到POVM的正确性的探讨.md`
- `docs/32_各povm的维度.md`
- `docs/33(已过时)_各povm的维度讨论2.md`

---

## 1) 现状树（拷自 README）

```
total_simulation.py               # CLI 主入口（参数解析 + 任务调度）
├── _parse_run_params()           # 解析 CLI 参数
└── main()                        # 调度入口（内部含汇总/统计辅助函数）

atom_sim/
├── __init__.py                    # 包导出
│   ├── MPSState
│   └── run_dual_atom_emission
│
├── core/                          # 数值核心层
│   ├── __init__.py
│   └── mps.py                     # MPSState 容器（TeNPy 后端）
│       ├── class MPSState
│       │   ├── __init__()
│       │   ├── _apply_two_site_op_local()
│       │   ├── apply_bond_op()
│       │   ├── apply_kraus_one_site()
│       │   ├── apply_one_site_gate()
│       │   ├── swap_sites()
│       │   ├── get_reduced_density()
│       │   ├── chi
│       │   ├── norm()
│       │   ├── get_bond_dimensions()
│       │   ├── copy()
│       │   └── __repr__()
│
├── hilbert/                       # 希尔伯特空间层（物理抽象）
│   ├── __init__.py
│   ├── basis.py                   # 空间定义和张量积
│   │   ├── class SubSpace          # 单子空间（780, 1517, atom）
│   │   ├── class ProductSpace      # 张量积空间 [s1, s2, ...]
│   │
│   └── operators.py               # 基本算符工厂
│       ├── annihilation_op()      # 湮灭算符 a[i]（复用实现）
│       ├── creation_op()          # 产生算符 a^†[i]
│       └── atom_transition()      # S_+, S_-（原子跃迁算符）
│
├── physics/                       # 物理过程层（门和通道）
│   ├── __init__.py
│   ├── gates.py                   # 所有酉门工厂
│   │   ├── qfc_gate()             # U_qfc: 780→1517 频率转换 (5x5)
│   │   ├── bs_gate_6d()           # U_BS: 50/50 分束器 (36x36, 仅1517nm)
│   │   ├── _bs_gate_1517()
│   │   ├── jones_gate()           # U_pol: 琼斯旋转 (6x6)
│   │   └── emission_gate()        # U_emit: 原子-光子耦合 (20x20)
│   │
│   └── channels.py                # 所有 Kraus 通道
│       ├── loss_channel_both_subspaces() # 780+1517 联合损耗
│       ├── loss_channel_1517_single_photon() # 1517nm 单光子损耗（3D）
│       └── FiberChannelParams            # 光纤漂移模型（琼斯+损耗）
│
├── simulation/                    # 仿真流程层（编排）
│   ├── __init__.py
│   ├── trajectory.py              # 单轨迹执行
│   │   ├── EmissionResult
│   │   ├── run_dual_atom_emission()
│   │   ├── _print_header()
│   │   ├── _print_footer()
│   │   └── apply_fiber_channel()
│   │
│   └── detection.py               # 探测和分析
│       ├── DetectionEvent
│       ├── TwoPhotonDetectionResult
│       ├── SuccessEnumerationResult
│       ├── _order_two_port_detectors()
│       ├── build_detection_effects_6d()
│       ├── run_detection_pipeline()     # POVM 枚举 + 抽样
│       ├── extract_spin_state()
│       ├── compute_fidelity_with_bell()
│
├── visualization/                 # 可视化层
│   ├── __init__.py
│   └── wavepacket.py              # 波包可视化
│       ├── _is_headless()
│       ├── _maybe_show()
│       ├── _get_bin6_state_labels()
│       ├── _get_bin5_state_labels()
│       ├── _get_bin3_state_labels()
│       ├── _infer_first_bin_site()
│       ├── _validate_bin_rho_traces()
│       ├── plot_dual_arm_heatmap()       # 绘制双臂热图
│
├── docs/                          # 设计文档与讨论
└── thesis/                        # 论文材料

outputs/                           # 仿真输出目录（已 gitignore）
└── <YYYYMMDD_HHMM>/               # 时间戳输出文件夹
    ├── results/
    │   └── result_<task_id>/
    │       ├── meta.json
    │       ├── raw/               # clicks.json / 调试输出等
    │       └── plots/             # after_emission/after_qfc 等热图
    ├── summary/
    │   ├── hom_trials.csv
    │   ├── hom_summary.csv
    │   ├── sim_summary.csv
    │   └── server_done.flag
    ├── tasks/
    │   ├── pending/
    │   ├── inprogress/
    │   └── done/
    └── heartbeat/
```

---

## 2) 目标树（改造后，细化到函数）

```
total_simulation.py
├── _parse_run_params()
├── _build_task_list()
├── _write_summary()
└── main()

atom_sim/
├── hilbert/
│   ├── basis.py
│   │   ├── class SubSpace
│   │   ├── class ProductSpace
│   └── operators.py
│       ├── annihilation_op()          # 5D 版本（含 780 / 1517 单光子）
│       ├── creation_op()
│       └── atom_transition()
│
├── physics/
│   ├── gates.py
│   │   ├── qfc_gate()                 # 改为 5×5
│   │   ├── bs_gate_6d()               # 保留 36×36（测量端共轭）
│   │   ├── _bs_gate_1517()
│   │   ├── jones_gate()               # 6×6（测量端）
│   │   └── emission_gate()
│   └── channels.py
│       ├── loss_channel_both_subspaces() # 改为 5D 版 Kraus（780/1517 单光子）
│       ├── loss_channel_1517_single_photon() # 3D 版（测量端）
│       └── class FiberChannelParams
│
├── simulation/
│   ├── trajectory.py
│   │   ├── class EmissionResult
│   │   ├── run_dual_atom_emission()   # 发射仍是态端唯一演化
│   │   ├── apply_fiber_channel()      # 只采样参数（U_A/U_B/eta/phase）
│   │   ├── _print_header/_footer
│   │   └── (删除 apply_780_filter / project_to_1517)
│   └── detection.py
│       ├── DetectionEvent / TwoPhotonDetectionResult / SuccessEnumerationResult
│       ├── build_detection_effects_6d()   # 现有保留
│       ├── run_detection_pipeline()       # 输入 bin_dim=5
│       ├── _project_6d_to_3d()
│       ├── _embed_3d_to_5d()
│
├── visualization/
│   └── wavepacket.py
│       ├── _get_bin6_state_labels()
│       ├── _get_bin5_state_labels()
│       ├── _get_bin3_state_labels()
│       ├── _infer_first_bin_site()
│       ├── _validate_bin_rho_traces()
│       ├── plot_dual_arm_heatmap()        # 局域重建热图
│
├── experiment/
│   ├── single_run.py
│   │   ├── _run_single_trial()
│   │   ├── _run_single_simulation_core()  # 仅传 fiber_sample / bs_unitary
│   │   └── _run_single_simulation_task()
│   └── hom.py
│       ├── _run_hom_run()                 # 仅传 fiber_sample / bs_unitary
│       └── parse_hom_cli()
│
└── docs/
    └── 31(计划)_5D大改_推POVM.md
```

> 关键一致性：QFC/BS 全部在测量端处理；态端不再保留 `apply_qfc/apply_bs` 占位接口。

---

## 3) 必须并入 POVM 的算符清单 + 方法论（含维度对齐）

### 3.0 关键前提（不满足则 36→9→25 不成立）

- **单光子截断**：每臂每个 bin 最多 1 光子（发射门与 QFC 不产生同臂双光子）。  
  若未来允许同臂双光子，BS 输入端也必须扩到 6D（甚至更高）。
- **无跨 bin 门**：BS/光纤/滤波/QFC 都是逐 bin（或逐 bin-pair）作用。  
  若加入色散/跨 bin 卷积，effect 会变成跨多个 bin 的 MPO（复杂度从态端迁移到算符端，而不会“消失”）。

### 3.0.1 三套 Hilbert 空间（闭环类型检查起点）

- **5D（态端 bin）**：`{vac, H780, V780, H1517, V1517}` → 双臂 25D  
  用于保留 QFC 失败分支（780 仍在账本里）。
- **3D（BS 前 telecom 单光子子空间）**：`{vac, H, V}` → 双臂 9D  
  用于 BS 输入端可达子空间（单光子截断）。
- **6D（BS 后/探测端）**：`{vac, H, V, 2H, 2V, HV}` → 双端口 36D  
  必须能容纳 bunching（同端口 2 光子）。

### 3.1 统一的 5D 基序（单臂单 bin）

```
|0> = |vac>
|1> = |H_780>
|2> = |V_780>
|3> = |H_1517>
|4> = |V_1517>
```

- 5D 仅含单光子（QFC/光纤/BS 推入 POVM 的前提）。
- 6D 只在测量端存在：`{vac, H, V, 2H, 2V, HV}`。

### 3.2 维度表（单端口 & 双端口）

| 阶段 | 单端口维度 | 双端口维度 | 说明 |
|---|---:|---:|---|
| MPS 输入 | 5 | 25 | 5D bin（含 780/1517 单光子） |
| QFC 后 | 5 | 25 | QFC 在 5D 内旋转 |
| 780 过滤后 | 5 | 25 | 780→vac 的 Kraus（仍在 5D） |
| 光纤后（BS 前） | 3 | 9 | 仅 1517 单光子 + vac（可投影到 3D） |
| BS 输出端 | 6 | 36 | 允许 2 光子（bunching） |

> 备注：`324 = 18×18` 只出现在“仍保留 18D bin”的旧路线；  
> 本计划明确采用 5D bin，故链条为 **36→9→25**。

### 3.2.1 “BS 用 36×36，但输入只有 3D/5D” 的闭环说明（来自文档32）

**核心结论：**
- BS 的物理 U 必须是 **36×36**（因为输出端需要 2 光子态）。
- 但 **输入端在 BS 之前** 只需要 3D（`vac,H,V`）或 5D（`vac,H780,V780,H1517,V1517`）。
- 把 BS 推到 POVM 后，**先对 36×36 的 effect 做共轭，再用投影把它压回输入子空间**。

**输入 3D 的投影闭环：**
- 单端口投影：`P_3←6 = [I_3, 0]`（取 `{vac,H,V}`），形状 `(3×6)`
- 双端口投影：`Π_3←6 = P_3←6 ⊗ P_3←6`，形状 `(9×36)`
- 压缩：`E_3 = Π_3←6 · (U_BS^† E_out U_BS) · Π_3←6^†` → `9×9`

**输入 5D 的嵌入闭环：**
- 单端口嵌入：`P_5←3` 把 `{vac,H,V}` 放到 5D 的 `{0,3,4}`，形状 `(5×3)`
- 双端口嵌入：`Π_5←3 = P_5←3 ⊗ P_5←3`，形状 `(25×9)`
- 扩展：`E_5 = Π_5←3 · E_3 · Π_5←3^†` → `25×25`

> 备注：  
> - `P_3←6` 取 `SUBSPACE_1517 = {vac,H,V,2H,2V,HV}` 的前三个基（索引 0,1,2）。  
> - `P_5←3` 把 `{vac,H,V}` 嵌入 5D 基 `{vac,H780,V780,H1517,V1517}` 的索引 `{0,3,4}`。  
> - 若矩阵为实数选择矩阵，`^T` 与 `^†` 等价，但公式统一用 `^†`。

### 3.3 推入 POVM 的算符清单（顺序必须正确）

> 以下 **全部来自当前代码**（函数名必须一致保留），并标注矩阵维度。  
> 这些算符全部进入 effect 的对偶映射；**态端不再 apply**。  
> **Heisenberg 端的推进顺序与 Schrödinger 正向相反。**

| 过程 | 代码函数 | 作用域 | 单端口维度 | 双端口维度 | 推入方式 |
|---|---|---|---:|---:|---|
| 探测 POVM | `build_detection_effects_6d()` | 6D | 6×6 | 36×36 | **起点**（直接构造） |
| BS | `bs_gate_6d()` | 6D | 6×6 | 36×36 | `E ← U_BS^† E U_BS` |
| 光纤 Jones/相位 | `jones_gate()`（3D 版：`diag(1,U_2×2)` 或由 6D 投影） | 3D | 3×3 | 9×9 | `E ← U^† E U` |
| 光纤损耗 | `loss_channel_1517_single_photon()` | 3D | 3×3 | 9×9 | `E ← Σ K^† E K` |
| 780 过滤 | `loss_channel_both_subspaces()` | 5D | 5×5 | 25×25 | `E ← Σ K^† E K` |
| QFC | `qfc_gate()` | 5D | 5×5 | 25×25 | `E ← U_qfc^† E U_qfc` |

> 检测器效率/暗计数已包含在 `build_detection_effects_6d()` 中，不再额外推。

---

## 4) 维度对齐与“算符推入”计算（带维度校验）

### 4.1 基本投影/嵌入矩阵（单端口）

- **6D → 3D 投影**：`P_3←6` 取 `{vac,H,V}`
  - 形状：`(3×6)`
- **3D → 5D 嵌入**：`P_5←3` 把 `{vac,H,V}` 放到 5D 的 `{0,3,4}`
  - 形状：`(5×3)`

### 4.2 双端口投影/嵌入

- `Π_3←6 = P_3←6 ⊗ P_3←6` → 形状 `(9×36)`
- `Π_5←3 = P_5←3 ⊗ P_5←3` → 形状 `(25×9)`

### 4.3 完整链条（维度逐步对齐，编译器式检查）

**输入**：`E_out` (36×36) 来自探测器 POVM

```
E_bs  = U_BS^† · E_out · U_BS           # 36×36
E_3   = Π_3←6 · E_bs · Π_3←6^†          # (9×36)(36×36)(36×9) = 9×9
E_f   = Λ_fiber^†(E_3)                  # 9×9
E_5   = Π_5←3 · E_f · Π_5←3^†           # (25×9)(9×9)(9×25) = 25×25
E_fil = Λ_filter^†(E_5)                 # 25×25
E_in  = U_qfc^† · E_fil · U_qfc         # 25×25
```

> 说明：  
> - `U_qfc` 在此指双臂算符 `U_qfc ⊗ U_qfc`（25×25）。  
> - `Λ_fiber^†` 与 `Λ_filter^†` 均是双臂对偶映射（A/B 两臂张量积）。  
> - 投影到 3D 的合法性来自“BS 前每臂最多 1 光子”的物理约束（文档32 已论证）。

**输出**：`E_in` 与 MPS 的 5D bin（单端口 5D、双端口 25）直接收缩。

> 对应概率与后验原子态：  
> `p = Tr[(I_atoms ⊗ E_in) ρ]`，  
> `ρ_atoms' = Tr_bins[(I_atoms ⊗ E_in) ρ] / p`（与现有 extract_spin_state 语义一致）。

> 若未来加入“跨 bin 色散/卷积”，该步骤不再是局域 9×9/25×25，  
> 而会变成跨 bin 的 MPO（复杂度不会消失，只会从 state 迁移到 operator）。

---

## 5) 5D QFC / 过滤 / 光纤信道的构造约束

### 5.1 5D QFC（单端口）

仅在 `|H_780> ↔ |H_1517>` 与 `|V_780> ↔ |V_1517>` 子空间旋转：

```
U_qfc = |vac><vac| ⊕ R_H(θ_H) ⊕ R_V(θ_V)
R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]
```

### 5.2 780 过滤（单端口 Kraus）

- 目标：把 `|H_780>, |V_780>` 完全损耗到 `|vac>`
- Kraus 示例（3 个）：
  - `K0 = |vac><vac| + |H_1517><H_1517| + |V_1517><V_1517|`
  - `K1 = |vac><H_780|`
  - `K2 = |vac><V_780|`
- 组合到 5D（保持 5×5）

### 5.3 光纤信道（单端口 3D）

- 建模域：`{|vac>, |H_1517>, |V_1517>}`
- Jones 旋转：`U_J(2×2)` 嵌入到 3D：`diag(1, U_J)`
- 损耗：用 3D Kraus（H/V 透过率，vac 保持）

---

## 6) 热图的“局域重建”方案（替代全局 after_* 态）

### after_fiber
- 取每个 bin 的 **1-site RDM**（5×5）
- 施加 `U_qfc` + `Λ_filter` + `Λ_fiber` 的 Schrödinger 端变换
- 读出 `{vac, H1517, V1517}` 的概率（或 ⟨n⟩）

### after_BS
- 取每个 bin 的 **2-site RDM**（25×25）
- 局域执行：`U_qfc` + `Λ_filter` + `Λ_fiber` → 得到 3D×3D
- 嵌入到 6D×6D，再施加 `U_BS`
- 读端口 1/2 的 `{vac,H,V,2H,2V,HV}` 分布

---

## 7) 关注点分离（必须守住的边界）

- `simulation/trajectory.py`：只做“发射态演化 + 采样参数”；不承载 POVM 计算。  
- `simulation/detection.py`：只做“测量/统计/收缩”；不采样光纤/BS 参数（由 trajectory 传入）。  
- `visualization/wavepacket.py`：只做“局域重建 + 画图”；不复用 detection 的抽样逻辑。  
- `experiment/*`：只组织实验流程；不直接拼 POVM 链条（统一交给 detection）。  
- `total_simulation.py`：只做 CLI/队列调度/任务划分；不承载物理链路细节。  

---

## 8) 模块改动清单（详细）

### 7.1 `hilbert/basis.py`
- `BIN_SPACE` 由 18D 改为 5D（新基序固定）。
- 保留 780/1517 子空间定义（3D/6D），用于测量端。

### 7.2 `physics/gates.py`
- `qfc_gate()` → 5×5 构造（只在 780/1517 的 H/V 子空间旋转）。
- `bs_gate_6d()` 保留（测量端共轭）。
- `jones_gate()` 保留 6×6（测量端/二光子端口），并支持 3×3 版用于 effect 链条（或在 detection 中用投影得到 3×3）。

### 7.3 `physics/channels.py`
- `loss_channel_both_subspaces()` 改成 5D 版 Kraus（780/1517 单光子）。
- `loss_channel_1517_single_photon()` 提供 3D 损耗 Kraus，直接用于 effect 链条。

### 7.4 `simulation/trajectory.py`
- 发射仍是唯一的态端演化。
- `apply_qfc()` 删除：QFC 完全在测量端 effect 中处理。
- `apply_bs()` 删除：BS 完全在测量端 effect 中处理。
- `apply_fiber_channel()` 只采样光纤参数（Jones/eta/phase），不修改 MPS。
- 删除 `apply_780_filter()` / `project_to_1517()`。

### 7.5 `simulation/detection.py`
- `run_detection_pipeline()` 接收 `bin_dim=5`。
- `_build_effect_chain_5d()` 统一实现 36→9→25 的 effect 链条。
- 仍保留 `effects_all/effects_true` 拆分逻辑。

### 7.6 `visualization/wavepacket.py`
- 删除 18D 专用分支（`telecom_ops_bin18`, `_get_bin18_state_labels` 等）。
- 新增 5D 标签与局域重建函数。

### 7.7 `single_run.py` / `hom.py`
- 传入 `fiber_sample` 给 detection。
- p_arrive 语义调整（不再直接由 MPS 统计 1517 光子数）。

---

## 9) 实施顺序（建议）

1) **basis / gates / channels**（建立 5D 基与通道）
2) **trajectory**（停用态端 QFC/滤波/损耗/BS）
3) **detection**（完整 effect 链条与维度映射）
4) **visualization**（局域重建热图）
5) **single_run / hom / total_simulation**（参数接入 + 输出字段修正）
6) **README / docs**（同步结构与流程描述）

---

## 10) 清理原则（防残留）

- 删除所有 18D 专用逻辑、断言与注释。
- 删除 apply_780_filter / project_to_1517 的引用与文档说明。
- 旧功能改造时保留函数名，不引入 new_ 前缀。

---

## 11) 风险点与验证

- 5D/3D/6D 映射是否一致（尤其是基序与投影矩阵）。
- p_arrive 的定义是否需改成“effect 侧统计”。
- HOM 统计是否仍匹配（p_arrive / coinc_rate 语义）。
