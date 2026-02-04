# 量子仿真：中性原子量子接口

中性原子量子纠缠协议的时间仓MPS仿真。

## 项目结构

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
│   │   ├── get_bin_space()
│   │   └── get_system_space()
│   │
│   └── operators.py               # 基本算符工厂
│       ├── annihilation_op()      # 湮灭算符 a[i]（复用实现）
│       ├── creation_op()          # 产生算符 a^†[i]
│       └── atom_transition()      # S_+, S_-（原子跃迁算符）
│
├── physics/                       # 物理过程层（门和通道）
│   ├── __init__.py
│   ├── gates.py                   # 所有酉门工厂
│   │   ├── qfc_gate()             # U_qfc: 780↔1517 频率转换 (5x5)
│   │   ├── bs_gate_6d()           # U_BS: 50/50 分束器 (36x36, 仅1517nm)
│   │   ├── _bs_gate_1517()
│   │   ├── jones_gate()           # U_pol: 琼斯旋转 (6x6)
│   │   └── emission_gate()        # U_emit: 原子-光子耦合 (20x20)
│   │
│   └── channels.py                # 所有 Kraus 通道
│       ├── loss_channel_both_subspaces() # 5D bin 联合损耗
│       ├── loss_channel_780_general()    # 780nm 损耗
│       ├── loss_channel_1517_raw()       # 1517nm 振幅阻尼（6D）
│       └── FiberChannelParams            # 光纤漂移模型（琼斯+损耗）
│
├── simulation/                    # 仿真流程层（编排）
│   ├── __init__.py
│   ├── trajectory.py              # 单轨迹执行
│   │   ├── EmissionResult
│   │   ├── run_dual_atom_emission()
│   │   ├── apply_qfc()            # Heisenberg 端口占位（不改态）
│   │   ├── apply_bs()             # Heisenberg 端口占位（不改态）
│   │   ├── _print_header()
│   │   ├── _print_progress()
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
│       └── compute_photon_statistics()
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
│       └── plot_cross_bin_joint_heatmap()# 跨 bin 联合分布热图
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

## 层次职责

| 层 | 职责 | 不关心 |
|-----|------|--------|
| `core/mps.py` | 张量网络存储、局域 TEBD 更新、SVD | 物理意义 |
| `hilbert/` | 线性代数：空间、基、算符 | 门做什么 |
| `physics/` | 物理：门矩阵、Kraus 通道 | MPS 更新 |
| `simulation/` | 编排：调用顺序、条件 | 矩阵如何计算 |
| `visualization/` | 结果可视化、数据提取 | - |

## 运行模式与任务队列

### 物理模式（task_type / mode）
- **SIM**：单次/多次仿真（输出点击记录、成功率、保真度等）
- **HOM**：HOM 扫描（输出 `hom_trials.csv` / `hom_summary.csv`）
- **SUMMARY**：内部汇总任务（不是 CLI 模式，由 worker 执行）

### 运行角色（role）
- **server**：生成任务、监控进度、归档输出
- **worker**：抢任务执行（SIM/HOM/SUMMARY）
- **both**：本机同时承担 server + worker

### 常用 CLI 选项（片段）
```
--role server|worker|both
--task-type SIM|HOM
--run-id <id>                 # 不传则自动选择最小可用 id
--queue-root <path>           # 默认 ./queue
--runs N --shots M
--plot-all                    # 每个 run 都画图
--no-plot                     # 禁止绘图（覆盖 plot-all）
```

## 数据流

```
simulation/trajectory.py → physics/gates.py → hilbert/basis.py → numpy 矩阵
                                               ↓
simulation/trajectory.py → core/mps.py → 张量网络更新（仅局域！）
```

## 核心设计原则

### 1. 非幺正操作不用 `apply_local_op`
所有 Kraus 和测量操作必须使用局域 theta + SVD 更新以避免 canonical sweep。使用：
- `apply_bond_op(i, op)` 用于双格点酉门
- `apply_kraus_one_site(i, {Kμ}, rng)` 用于单格点 Kraus 采样
- `apply_kraus_one_site_fixed(i, Kμ)` 用于固定分支（如“无损耗”）

### 2. 格点类型：有限维格点，非 BosonSite
使用自定义的 `FiniteDimSite(d)` 配合算符字典，而非 TeNPy 的 `BosonSite`（语义不兼容）。

### 3. 密度矩阵提取
始终使用 `get_reduced_density([i])`，绝不直接收缩 `_B[i]`。

## 物理模型

### 原子能级（4D）
- `|e>`: 5P_{3/2}, F'=0, m_F=0 （激发态）
- `|0>`: 5S_{1/2}, F=1, m_F=+1 （基态）
- `|1>`: 5S_{1/2}, F=1, m_F=-1 （基态）
- `|u>`: 5S_{1/2}, F=1, m_F=0 （基态）

### 选择定则
- `|e> → |0>`: Δm = +1 → σ+ 光子（右圆偏振）
- `|e> → |1>`: Δm = -1 → σ- 光子（左圆偏振）

### 光子子空间
- **780nm**: 3D `{|vac>, |H>, |V>}` （单光子截断）
- **1517nm（测量端）**: 6D `{|vac>, |H>, |V>, |2H>, |2V>, |HV>}` （双光子截断）
- **MPS bin 可达子空间**: 5D `{|vac>, |H_780>, |V_780>, |H_1517>, |V_1517>}`

### 希尔伯特空间分解

```
系统格点: H_S = H_atom_A(4D) ⊗ H_atom_B(4D) = 16D
Bin 格点: H_bin = span{|vac>, |H_780>, |V_780>, |H_1517>, |V_1517>} = 5D
```

**重要**：原子仅在系统格点中，不在 bin 格点中。

### MPS 链布局

**发射后（SWAP conveyor belt 完成）：**
```
atomA(4D) - atomB(4D) - A1(5D) - B1(5D) - A2(5D) - B2(5D) - ... - AN(5D) - BN(5D)
```

相邻的 (A_n, B_n) 对用于在测量端构造 BS+POVM 的联合效果。

## 仿真流程

```python
# (1) 发射：SWAP conveyor belt 协议
result = run_dual_atom_emission(n_bins=100, ...)
# 结果：原子在末尾，bins 包含 780nm 光子

# (2) QFC：780nm ↔ 1517nm 频率转换
apply_qfc(mps, n_bins, theta_H=π/4, theta_V=π/4)
# 注：QFC/过滤/光纤/BS 已整体推入 POVM（Heisenberg 端口），
#     apply_qfc 仅记录参数并输出日志，不改动 MPS。

# (3) 光纤采样（不改态）
mps, fiber_sample, _ = apply_fiber_channel(...)
# 注：光纤的 Jones/损耗/相位漂移在 POVM 端使用 fiber_sample 重建。

# (4) 分束器并入测量端：构造 U_BS^† E U_BS
# 结果：在不显式作用 BS 的情况下获取端口点击分布

# (5) 探测：POVM 枚举 + 抽样（QFC/过滤/光纤/BS 都已并入测量端）
pipeline = run_detection_pipeline(
    mps, n_bins, eta_det=eta_det, p_dark=p_noise, bs_unitary=bs_gate_6d()
)
# 结果：点击事件列表 / 成功率 / 保真度

# (6) BSM 宣告：检查成功模式
# Ψ-: (H1, V2) 或 (V1, H2) - 跨端口不同偏振（反聚束）
# Ψ+: (H1, V1) 或 (H2, V2) - 同端口不同偏振（聚束）
```

### 点击记录格式

SIM/HOM 产生的点击记录（`raw/clicks.json` 与 `sim_summary.csv`）格式为：

```
[(detector, bin_index, is_dark), ...]
```

- `detector`: `H1` / `V1` / `H2` / `V2`
- `bin_index`: 点击发生的时间仓索引
- `is_dark`: 是否为暗记数触发的点击（True/False）

### 成功率字段（SIM / sim_summary.csv）

- `p_success_abs`：每次尝试的**总成功率**（含暗计数）
- `p_success_true_abs`：成功中“纯真实点击”的部分
- `p_success_false_abs`：成功中“含暗计数”的部分（=`p_success_abs - p_success_true_abs`）
- `p_success_true_given_arrival`：在“两光子到达”条件下的真实成功率（`p_success_true_abs / p_arrive`）
- `p_success_no_dark_abs`：暗计数关掉时的成功率基线（若枚举 no-dark）

### FiberChannelParams

模拟真实光纤传输的随机漂移：

```python
fiber_params = FiberChannelParams(
    polarization_model="perturb",  # "haar", "perturb", 或 "euler"
    polarization_sigma=0.1,        # 旋转角度标准差 (rad)
    eta_mean=0.57,                 # 平均透过率
    eta_std=0.02,                  # 透过率波动
    pdl_sigma=0.02,                # 小PDL：H/V透过率相对差异
    phase_drift_std=0.2,           # 臂间相位漂移 (rad)
)

# 为单次轨迹采样参数
U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase = fiber_params.sample_all(rng)
```

## 波包提取

波包形状编码在时间仓占据概率中：

```python
# 提取强度包络
p_n = <N_H_n + N_V_n> 对每个 bin n

# 提取复振幅（用于 HOM 可见度）
xi_n^H = <1_H_n|psi>, xi_n^V = <1_V_n|psi>

# 模式重叠（决定 HOM 可见度）
M = |sum_n (xi_A_n^H* xi_B_n^H + xi_A_n^V* xi_B_n^V)|^2
```

## 依赖项

- `numpy` - 数组操作
- `physics-tenpy` - 张量网络后端
- `matplotlib` - 可视化（热图/联合分布）

## 参考资料

详见 `docs/` 目录中的详细规范（文件名含编号与状态标记）：
- `3(部分完成_差PMD)_总设计图纸.md` - 整体架构
- `5(已过时)_逐行流水表.md` - 早期执行流程（已淘汰）
- `4(部分完成_差参数)_要模拟的对象与输出.md` - 实现规范
- `6(部分完成_部分废案_差站点)_有关空间排列与直积构造相关修改建议.md` - 设计修正与历史方案
