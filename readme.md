# 量子仿真：中性原子量子接口

中性原子量子纠缠协议的时间仓MPS仿真。

## 项目结构

```
total_simulation.py               # CLI 主入口（全流程仿真 + 汇总输出）
├── _parse_run_params()
├── save_debug_info()
├── _append_click_summary()
├── _init_stats()
├── _merge_stats()
├── _format_counter()
├── _write_csv_header()
├── _append_csv_row()
├── _format_metric()
├── _format_stat()
├── _file_lock()
├── _init_combined_summary()
├── _finalize_combined_summary()
├── _write_extra_data()
├── _write_success_metrics_detail()
├── _init_success_metrics_accumulator()
├── _accumulate_success_metrics()
├── _finalize_success_metrics()
├── _run_single_simulation_core()
├── _run_single_simulation()
├── _run_single_simulation_task()
└── main()

atom_sim/
├── __init__.py                    # 包导出
│   ├── MPSState
│   ├── TimeGrid
│   └── run_dual_atom_emission
│
├── time_grid.py                   # 时间网格
│   └── TimeGrid
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
│       │   ├── apply_two_site_kraus()
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
│   │   ├── subspace_gate()         # 将子空间门嵌入到积空间
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
│   │   ├── qfc_gate()             # U_qfc: 780→1517 频率转换 (18x18)
│   │   ├── bs_gate_6d()           # U_BS: 50/50 分束器 (36x36, 仅1517nm)
│   │   ├── _bs_gate_1517()
│   │   ├── jones_gate()           # U_pol: 琼斯旋转 (6x6)
│   │   └── emission_gate()        # U_emit: 原子-光子耦合 (54x54)
│   │
│   └── channels.py                # 所有 Kraus 通道
│       ├── loss_channel_both_subspaces() # 780+1517 联合损耗
│       ├── loss_channel_780_general()    # 780nm 损耗
│       ├── loss_channel_1517_raw()       # 1517nm 振幅阻尼（6D）
│       └── FiberChannelParams            # 光纤漂移模型（琼斯+损耗）
│
├── simulation/                    # 仿真流程层（编排）
│   ├── __init__.py
│   ├── trajectory.py              # 单轨迹执行
│   │   ├── EmissionResult
│   │   ├── run_dual_atom_emission()
│   │   ├── apply_qfc()
│   │   ├── apply_780_filter()
│   │   ├── project_to_1517()
│   │   ├── apply_bs()
│   │   ├── _print_header()
│   │   ├── _print_progress()
│   │   ├── _print_footer()
│   │   └── apply_fiber_channel()
│   │
│   └── detection.py               # 探测和分析
│       ├── DetectionEvent
│       ├── TwoPhotonDetectionResult
│       ├── SuccessEnumerationResult
│       ├── _order_detectors()
│       ├── _order_two_port_detectors()
│       ├── _split_with_dark()
│       ├── _build_port_kraus_entries_6d()
│       ├── _build_detection_kraus()
│       ├── build_detection_kraus_6d()
│       ├── run_two_photon_detection()   # 量子跃迁采样
│       ├── extract_spin_state()
│       ├── check_bsm_success()
│       ├── compute_fidelity_with_bell()
│       ├── _infer_bin_start()
│       ├── _get_bin_sites()
│       ├── _build_photon_number_projectors()
│       ├── compute_two_photon_arrival_prob()
│       ├── _build_detection_effects()
│       ├── _bell_projector_full()
│       ├── _prepare_grouped_mps_pairs()
│       ├── _apply_env_left()
│       ├── _apply_env_right()
│       ├── _build_left_envs()
│       ├── _build_right_envs()
│       ├── enumerate_success_events()   # POVM 枚举
│       ├── _compute_photon_statistics_global()
│       ├── _build_sum_mpo()
│       └── compute_photon_statistics()
│
├── visualization/                 # 可视化层
│   ├── __init__.py
│   └── wavepacket.py              # 波包可视化
│       ├── _is_headless()
│       ├── _maybe_show()
│       ├── _telecom_ops_1517()
│       ├── telecom_ops_bin18()
│       ├── _get_bin18_state_labels()
│       ├── _get_bin6_state_labels()
│       ├── _infer_first_bin_site()
│       ├── _validate_bin_rho_traces()
│       ├── plot_dual_arm_heatmap()       # 绘制双臂热图
│       └── plot_cross_bin_joint_heatmap()# 跨 bin 联合分布热图
│
├── docs/                          # 设计文档与讨论
├── notebooks/                     # 交互式实验
└── thesis/                        # 论文材料

outputs/                           # 仿真输出目录（已 gitignore）
└── <YYYYMMDD_HHMM>/               # 时间戳输出文件夹
    ├── runXXX_1_after_emission.png
    ├── runXXX_2_after_qfc.png
    ├── runXXX_3_after_fiber.png
    ├── runXXX_4_after_bs.png
    ├── runXXX_4b_after_bs_cross_bin_joint.png
    ├── runXXX_extra_data.txt
    └── all_clicks_summary.csv
```

## 层次职责

| 层 | 职责 | 不关心 |
|-----|------|--------|
| `core/mps.py` | 张量网络存储、局域 TEBD 更新、SVD | 物理意义 |
| `hilbert/` | 线性代数：空间、基、算符 | 门做什么 |
| `physics/` | 物理：门矩阵、Kraus 通道 | MPS 更新 |
| `time_grid.py` | 时间网格参数 | 计算 |
| `simulation/` | 编排：调用顺序、条件 | 矩阵如何计算 |
| `visualization/` | 结果可视化、数据提取 | - |

## 数据流

```
time_grid.py → physics/gates.py → hilbert/basis.py → numpy 矩阵
                                               ↓
simulation/trajectory.py → core/mps.py → 张量网络更新（仅局域！）
```

## 核心设计原则

### 1. 非幺正操作不用 `apply_local_op`
所有 Kraus 和测量操作必须使用局域 theta + SVD 更新以避免 canonical sweep。使用：
- `apply_bond_op(i, op)` 用于双格点门
- `apply_kraus_one_site(i, {Kμ}, rng)` 用于单格点 Kraus
- `apply_two_site_kraus(i, {Kμ}, rng)` 用于双格点测量

### 2. 格点类型：有限维格点，非 BosonSite
使用自定义的 `FiniteDimSite(d)` 配合算符字典，而非 TeNPy 的 `BosonSite`（语义不兼容）。

### 3. 密度矩阵提取
始终使用 `get_reduced_density([i])`，绝不直接收缩 `_B[i]`。

## 物理模型

### 原子能级（3D）
- `|e>`: 5P_{3/2}, F'=0, m_F=0 （激发态）
- `|0>`: 5S_{1/2}, F=1, m_F=+1 （基态）
- `|1>`: 5S_{1/2}, F=1, m_F=-1 （基态）

### 选择定则
- `|e> → |0>`: Δm = +1 → σ+ 光子（右圆偏振）
- `|e> → |1>`: Δm = -1 → σ- 光子（左圆偏振）

### 光子子空间
- **780nm**: 3D `{|vac>, |H>, |V>}` （单光子截断）
- **1517nm**: 6D `{|vac>, |H>, |V>, |2H>, |2V>, |HV>}` （双光子截断）

### 希尔伯特空间分解

```
系统格点: H_S = H_atom_A(3D) ⊗ H_atom_B(3D) = 9D
Bin 格点: H_bin = H_780(3D) ⊗ H_1517(6D) = 18D
```

**重要**：原子仅在系统格点中，不在 bin 格点中。

### MPS 链布局

**发射后（SWAP conveyor belt 完成）：**
```
A1(18D) - B1(18D) - A2(18D) - B2(18D) - ... - AN(18D) - BN(18D) - atomA(3D) - atomB(3D)
```

相邻的 (A_n, B_n) 对方便进行分束器和探测操作。

## 仿真流程

```python
# (1) 发射：SWAP conveyor belt 协议
result = run_dual_atom_emission(n_bins=100, ...)
# 结果：原子在末尾，bins 包含 780nm 光子

# (2) QFC：780nm → 1517nm 频率转换
apply_qfc(mps, n_bins, theta_H=π/4, theta_V=π/4)
# 结果：光子转换为通信波长

# (3) 780nm 滤波：移除未转换的光子
apply_780_filter(mps, n_bins)

# (4) 分束器：干涉 A_n 与 B_n
apply_bs(mps, n_bins)
# 结果：每个 bin 对的 HOM 干涉

# (5) 探测：on/off 光子探测
det_result = run_two_photon_detection(mps, n_bins, eta_det, rng)
# 结果：点击事件列表

# (6) BSM 宣告：检查成功模式
# Ψ+: (H1, V2) 或 (V1, H2) - 跨端口不同偏振（反聚束）
# Ψ-: (H1, V1) 或 (H2, V2) - 同端口不同偏振（聚束）
```

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

## 参考资料

详见 `docs/` 目录中的详细规范：
- `总设计图纸.md` - 整体架构
- `逐行流水表.md` - 详细执行流程
- `要模拟的对象与输出.md` - 实现规范
- `有关空间排列与直积构造相关修改建议.md` - 设计修正和说明
