# 量子仿真：中性原子量子接口

中性原子量子纠缠协议的时间仓MPS仿真。

## 项目结构

```
atom_sim/
├── __init__.py
│
├── core/                          # 数值核心层
│   ├── __init__.py
│   └── mps.py                     # MPSState 容器（TeNPy 后端）
│                                   # - apply_bond_op() : 双格点局域更新
│                                   # - apply_one_site_gate() : 单格点酉门
│                                   # - apply_kraus_one_site() : 单格点 Kraus 轨迹
│                                   # - apply_kraus_two_site() : 双格点 Kraus
│                                   # - swap_sites() : SWAP 门（交换两个格点）
│                                   # - get_reduced_density() : 约化密度矩阵
│                                   # - chi 属性 : 键维度列表
│
├── hilbert/                       # 希尔伯特空间层（物理抽象）
│   ├── __init__.py
│   ├── basis.py                   # 空间定义和张量积
│   │   ├── SubSpace               # 单子空间（780, 1517, atom）
│   │   ├── ProductSpace           # 张量积空间 [s1, s2, ...]
│   │   └── subspace_gate()        # 将子空间门嵌入到积空间
│   │
│   └── operators.py               # 基本算符工厂
│       ├── annihilation_op()      # 湮灭算符 a[i]（复用实现）
│       ├── creation_op()          # 产生算符 a^†[i]
│       ├── atom_transition()      # S_+, S_-（原子跃迁算符）
│       └── number_op()            # 粒子数算符 N = a^† a
│
├── physics/                       # 物理过程层（门和通道）
│   ├── __init__.py
│   ├── gates.py                   # 所有酉门工厂
│   │   ├── emission_gate()        # U_emit: 原子-光子耦合 (54x54)
│   │   ├── qfc_gate()             # U_qfc: 780→1517 频率转换 (18x18)
│   │   ├── bs_gate_6d()           # U_BS: 50/50 分束器 (36x36, 仅1517nm)
│   │   ├── jones_gate()           # U_pol: 琼斯旋转 (6x6)
│   │   ├── jones_gate_from_array() # 从 2x2 琼斯矩阵构造门
│   │   └── swap_gate()            # W_swap: SWAP 门 (324x324)
│   │
│   └── channels.py                # 所有 Kraus 通道
│       ├── loss_channel_1517()           # 1517nm 振幅阻尼
│       ├── loss_channel_both_subspaces() # 780+1517 联合损耗
│       ├── loss_channel_780_general()    # 780nm 损耗（内部）
│       ├── detection_channel()           # 单模 on/off POVM
│       ├── detection_povm_single_site()  # 每格点 H+V 探测（4结果）
│       ├── detection_channel_two_mode()  # 双端口探测（16结果）
│       ├── dephasing_channel()           # 原子退相位
│       └── FiberChannelParams            # 光纤漂移模型（琼斯+损耗）
│
├── config.py                      # 所有参数类
│   ├── TimeGrid                   # dt, N, t[n]
│   ├── EmitParams                 # gamma(t), Alpha 矩阵
│   ├── QFCParams                  # theta_H, theta_V
│   ├── FiberParams                # 琼斯矩阵, PMD 参数
│   └── DetParams                  # eta_det, p_dark, success_patterns
│
├── simulation/                    # 仿真流程层（编排）
│   ├── __init__.py
│   ├── trajectory.py              # 单轨迹执行
│   │   ├── TrajectoryRunner       # 主循环
│   │   │   ├── initialize_mps()
│   │   │   ├── run_emission()     # 发射流程
│   │   │   └── run_bin(n)         # 单个时间仓的完整流程
│   │   │
│   │   ├── EmissionResult         # 发射阶段结果容器
│   │   │
│   │   └── apply_* 函数:          # 统一处理接口
│   │       ├── apply_qfc()        # 对所有 bins 应用 QFC
│   │       ├── apply_780_filter() # 过滤未转换的 780nm 光子
│   │       ├── apply_jones()      # 对所有 bins 应用琼斯旋转
│   │       ├── apply_loss()       # 损耗通道（仅 1517）
│   │       ├── apply_loss_combined() # 损耗通道（780+1517）
│   │       ├── apply_fiber_channel() # 琼斯+损耗（随机采样）
│   │       └── apply_bs()         # 对所有 bin 对应用分束器
│   │
│   ├── simulator.py               # 多轨迹统计
│   │   └── run_simulation()       # 返回 p_succ ± stderr, F_cond ± stderr
│   │
│   └── detection.py               # 探测和分析
│       ├── run_two_photon_detection()   # 量子跃迁方法
│       ├── compute_photon_statistics()  # 光子统计
│       ├── compute_fidelity_with_bell() # Bell 态保真度
│       └── extract_spin_state()         # 提取原子自旋态
│
├── visualization/                 # 可视化层
│   ├── __init__.py
│   └── wavepacket.py              # 波包可视化
│       ├── telecom_ops_bin18()           # 1517nm 光场算符
│       ├── extract_wavepacket()          # 提取波包复振幅
│       ├── extract_intensity_envelope()  # 提取强度包络
│       ├── extract_single_photon_prob()  # 提取单光子概率
│       ├── plot_wavepacket()            # 绘制波包图
│       ├── plot_intensity_envelope()     # 绘制强度包络
│       ├── plot_single_photon_prob()     # 绘制单光子概率
│       ├── extract_bin_state_probabilities()  # 提取 bin 态概率
│       ├── plot_bin_state_heatmap()      # 绘制 bin 态热图
│       ├── plot_dual_arm_heatmap()       # 绘制双臂热图
│       ├── extract_bin_state_coherences()    # 提取 bin 态相干
│       ├── plot_dual_arm_heatmap_phase()     # 绘制相位感知热图
│       └── extract_first_order_coherence()   # 提取一阶相干函数
│
└── tests/                         # 单元测试（正确性检查）
    ├── test_bs_closure.py         # |1>|1> 保持在截断空间内
    ├── test_kraus_completeness.py # sum(K^† K) = I
    └── ...

outputs/                           # 仿真输出目录
└── <YYYYMMDD_HHMM>/               # 时间戳输出文件夹
    ├── 1_after_emission.png
    ├── 2_after_qfc.png
    └── 3_after_fiber.png
```

## 层次职责

| 层 | 职责 | 不关心 |
|-----|------|--------|
| `core/mps.py` | 张量网络存储、局域 TEBD 更新、SVD | 物理意义 |
| `hilbert/` | 线性代数：空间、基、算符 | 门做什么 |
| `physics/` | 物理：门矩阵、Kraus 通道 | MPS 更新 |
| `config.py` | 数据：参数存储 | 计算 |
| `simulation/` | 编排：调用顺序、条件 | 矩阵如何计算 |
| `visualization/` | 结果可视化、数据提取 | - |
| `tests/` | 正确性验证 | - |

## 数据流

```
config.py → physics/gates.py → hilbert/basis.py → numpy 矩阵
                                               ↓
simulation/trajectory.py → core/mps.py → 张量网络更新（仅局域！）
```

## 核心设计原则

### 1. 非幺正操作不用 `apply_local_op`
所有 Kraus 和测量操作必须使用局域 theta + SVD 更新以避免 canonical sweep。使用：
- `apply_bond_op(i, op)` 用于双格点门
- `apply_kraus_one_site(i, {Kμ}, rng)` 用于单格点 Kraus
- `apply_kraus_two_site(i, {Kμ}, rng)` 用于双格点测量

### 2. Bin 废弃机制
测量后，bins 被冻结（键维度=1）且不再被访问。使用 `finalize_bin_pair(n)` 确保线性复杂度。

### 3. 格点类型：有限维格点，非 BosonSite
使用自定义的 `FiniteDimSite(d)` 配合算符字典，而非 TeNPy 的 `BosonSite`（语义不兼容）。

### 4. 密度矩阵提取
始终使用 `get_rho_segment([i])`（TeNPy 的方法），绝不直接收缩 `_B[i]`。

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
    phase_drift_std=0.2,           # 臂间相位漂移 (rad)
)

# 为单次轨迹采样参数
U_A, U_B, eta, phase = fiber_params.sample_all(rng)
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
