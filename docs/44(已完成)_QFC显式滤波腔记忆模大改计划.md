# QFC 显式滤波腔记忆模大改计划（44号）


> 目标：在不保留旧路径/兼容分支的前提下，把 QFC 链路从“每 bin 独立的等效 POVM”升级为“含 1517 滤波腔记忆的严格模型”。  
> 原则：一次性替换旧实现，不做回退/降级/兼容；同功能函数优先原位改造，不引入 `*_legacy/*_new` 并存。

---

## 0) 已读范围（为保证计划闭环）

- `README.md`（当前项目 tree 与分层职责）
- `docs/31(已完成)_5D大改_推POVM.md`（计划文档风格模板）
- `docs/41_专家对于当前腔发射以及QFC物理流程的优化意见7.md`
- `docs/43_专家对QFC链路的优化意见.md`
- `atom_sim/physics/gates.py`
- `atom_sim/simulation/detection.py`
- `atom_sim/core/mps.py`
- `atom_sim/experiment/common.py`
- `atom_sim/simulation/trajectory.py`
- `atom_sim/visualization/wavepacket.py`
- `total_simulation.py`

---

## 1) 当前树（拷自 README）

```text
total_simulation.py               # CLI 主入口（参数解析 + 任务调度）
├── _parse_run_params()           # 解析 CLI 参数
├── _build_task_list()            # 生成 SIM/HOM/WINDOW/BSM/LENGTH/SUMMARY 任务
├── _run_server_monitor()         # server 进度监控与 ETA 显示
├── _run_worker_loop()            # worker 抢占任务并执行
├── _write_summary()              # 调用 experiment.summary 产出汇总 CSV
└── main()                        # 调度入口（内部含汇总/统计辅助函数）

atom_sim/
├── __init__.py                    # 包导出
│   ├── MPSState
│   └── run_dual_atom_emission
│
├── core/                          # 数值核心层
│   ├── __init__.py
│   └── mps.py                     # MPSState + 探测收缩引擎（TeNPy 后端）
│       ├── class MPSState
│       │   ├── __init__()
│       │   ├── _apply_two_site_op_local()
│       │   ├── apply_bond_op()
│       │   ├── apply_kraus_one_site()
│       │   ├── swap_sites()
│       │   ├── get_reduced_density()
│       │   ├── chi
│       │   ├── norm()
│       │   ├── get_bond_dimensions()
│       │   ├── copy()
│       │   └── __repr__()
│       ├── compute_joint_arrival_probabilities()
│       └── class DetectionContractionEngine
│
├── hilbert/                       # 希尔伯特空间层（物理抽象）
│   ├── __init__.py
│   ├── basis.py                   # 空间定义和张量积
│   │   ├── class SubSpace          # 单子空间（780, 1517, atom）
│   │   ├── embed_9_from_6()        # 6D->9D 标签嵌入
│   │   └── reduce_9d_effects_to_6d() # 9D effect 回投影到 6D
│   │
│   └── operators.py               # 基本算符工厂
│       ├── annihilation_op()      # 湮灭算符 a[i]（复用实现）
│       ├── creation_op()          # 产生算符 a^†[i]
│       └── atom_transition()      # S_+, S_-（原子跃迁算符）
│
├── physics/                       # 物理过程层（门和通道）
│   ├── __init__.py
│   ├── gates.py                   # 酉门 + 探测 effect 构造（底层）
│   │   ├── qfc_gate()             # U_qfc: 780↔1517 频率转换 (5x5)
│   │   ├── bs_gate_6d()           # U_BS: 可调混合角分束器 (36x36, 仅1517nm)
│   │   ├── emission_gate()        # U_emit: 原子-光子耦合 (20x20)
│   │   ├── build_detection_effects_6d()/9d()
│   │   ├── build_arrival_projectors_5d()
│   │   └── build_detection_effects_5d_by_bin()
│   │
│   └── channels.py                # 所有 Kraus 通道
│       ├── loss_channel_both_subspaces() # 5D bin 联合损耗
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
│   │   └── sample_fiber_realization()
│   │
│   └── detection.py               # 探测和分析
│       ├── DetectionEvent
│       ├── TwoClickRecord
│       ├── TwoPhotonDetectionResult
│       ├── SuccessEnumerationResult
│       ├── DetectionPipelineResult
│       ├── _order_two_port_detectors()  # 适配层
│       ├── run_detection_pipeline()     # 编排：到达统计 + 枚举 + 抽样
│       ├── compute_pauli_correlators_and_chsh() # 关联量与 CHSH
│       ├── run_detection_self_checks()  # POVM 完备性/一致性自检
│       ├── extract_qubit_state()
│       └── compute_fidelity_with_bell()
│
├── experiment/                    # 任务实现层（按 task_type 拆分）
│   ├── common.py                  # 跨任务共享配置/参数构造
│   │   ├── class SimConfig        # 总配置对象（run/emission/noise/detector/...）
│   │   ├── run_emission_to_bs()   # 发射→QFC参数→光纤参数→退相干→BS诊断hook
│   │   ├── _build_run_parameter_store() # 统一参数表（eta/噪声映射/窗口/预算）
│   │   └── _build_detection_kwargs()    # 探测端统一入参组装
│   ├── single_run.py              # SIM 任务核心（单 run 执行）
│   │   ├── _run_single_trial()    # 单次物理链路调用
│   │   └── _run_single_simulation_core() # SIM 主流程（枚举+抽样+落盘）
│   ├── hom.py                     # HOM 任务（tau 扫描）
│   │   ├── parse_hom_cli()        # HOM 参数解析
│   │   ├── _build_hom_tau_values()# 生成 tau 扫描序列
│   │   └── _run_hom_run()         # 单 tau 的 run 执行
│   ├── window_scan.py             # WINDOW_SCAN 任务（window 扫描）
│   │   └── run_window_scan_task() # 同 run 复用发射态，扫描 window_ns
│   ├── bsm_scan.py                # BSM_SCAN 任务（BS 误差扫描）
│   │   └── run_bsm_scan_task()    # 同 run 复用发射态，扫描 bs_theta
│   ├── length_scan.py             # LENGTH_SCAN 任务（length 扫描）
│   │   └── run_length_scan_task() # 按长度重跑链路，统计 event_rate_hz
│   └── summary.py                 # 汇总任务输出
│       └── write_summary()        # 输出 *_trials / *_runs / *_summary
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
```

---

## 2) 大改边界与硬约束

1. **不保留旧 QFC 路径**：删除“每 bin 独立 QFC+filter+bg OR-map”的旧算法路径。
2. **不做兼容/回退**：新参数生效后，旧参数字段直接删，不保留双分支。
3. **保持职责分离**：
   - 线代/门/通道在 `physics/*`；
   - 收缩与复杂度控制在 `core/mps.py`；
   - 流程编排在 `simulation/detection.py`；
   - CLI/参数在 `experiment/common.py` 与 `total_simulation.py`。
4. **保留现有发射主链**：12D emitter 发射链路不回退。

---

## 3) 新方案数学定义（$...$ / $$...$$）

## 3.1 QFC 本体：偏振分辨的频域线性光学

单偏振 $p\in\{H,V\}$，定义输入频率模 $a_{780,p},a_{1517,p}$。QFC 在强泵近似下是 SU(2) 旋转：

$$
\begin{bmatrix}
a'_{780,p}\\
a'_{1517,p}
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_p & -e^{-i\phi_p}\sin\theta_p\\
e^{i\phi_p}\sin\theta_p & \cos\theta_p
\end{bmatrix}
\begin{bmatrix}
a_{780,p}\\
a_{1517,p}
\end{bmatrix}
$$

其中 $\eta_{\mathrm{conv},p}=\sin^2\theta_p$。这一步仍是**无记忆**局域变换。

## 3.2 1517 滤波腔记忆：跨 bin 相关的来源

对每条臂引入滤波腔记忆模 $f_n$（单光子截断 3D：$|vac\rangle,|H\rangle,|V\rangle$）。

令时间步长 $\Delta t$、滤波腔 FWHM 为 $\Delta\nu_f$（Hz），对应幅度记忆系数：

$$
r = \exp\!\left(-\pi\Delta\nu_f\Delta t\right), \quad
t = \sqrt{1-r^2}
$$

单步记忆更新（每个偏振同式）：

$$
f_{n+1} = r f_n + t b_n,
\qquad
c_n = -t f_n + r b_n
$$

其中 $b_n$ 是第 $n$ 个 bin 的 1517 输入模，$c_n$ 是滤波后输出模。该递推等价于二体幺正：

$$
U_{f,b_n}=\exp\!\left[\vartheta\,(f^\dagger b_n - f b_n^\dagger)\right],
\quad \cos\vartheta=r,\ \sin\vartheta=t
$$

因此**不需要 bin-bin 门**，但会自然产生跨 bin 相关。

## 3.3 状态端显式记忆链（完全采纳文档43/45）

本计划不走“测量端 MPO 主路”，而是把滤波记忆直接建在状态端：

1. 在发射后的链上新增两条记忆模（A/B 各一个，3D：$|vac\rangle,|H\rangle,|V\rangle$）；
2. 每个 bin 依次与本臂记忆模作用二体门；
3. 记忆模内部损耗/退相干用 Kraus 在每步更新；
4. 最后只在探测端做“BS+探测器+暗计数”的 effect 收缩。

扩展态记作：

$$
|\Psi_0^{\mathrm{aug}}\rangle = |\Psi_{\mathrm{emit}}\rangle\otimes|vac\rangle_{f_A}\otimes|vac\rangle_{f_B}
$$

第 $n$ 步（bin index 从 1 到 $N$）更新：

$$
|\Psi_{n+1}^{\mathrm{aug}}\rangle =
\left(U_{f_A,b_{A,n}}\,U_{f_B,b_{B,n}}\,U_{\mathrm{qfc},A,n}\,U_{\mathrm{qfc},B,n}\right)
|\Psi_n^{\mathrm{aug}}\rangle
$$

并施加记忆模噪声通道：

$$
\rho_{n+1}^{\mathrm{aug}} =
\left(\Lambda_{f_A}^{\mathrm{mem}}\otimes\Lambda_{f_B}^{\mathrm{mem}}\right)
\left(U_n\rho_n^{\mathrm{aug}}U_n^\dagger\right)
$$

最终记录概率：

$$
p(\text{record}) = \langle\Psi_{\mathrm{out}}^{\mathrm{aug}}|E_{\mathrm{det}}(\text{record})|\Psi_{\mathrm{out}}^{\mathrm{aug}}\rangle
$$

其中 $E_{\mathrm{det}}$ 仅承载探测端链路（BS、探测效率、本征暗计数、可区分度混合等），不再承载 QFC/滤波记忆。

## 3.4 QFC 背景：由谱密度驱动并注入记忆模

将 QFC 噪声参数改为谱密度口径（每臂）：$S_{\mathrm{bg}}$（cps/MHz）。

每臂有效背景计数率：

$$
R_{\mathrm{bg}} \approx
S_{\mathrm{bg}}\,\Delta\nu_f\,\eta_{\mathrm{filter}}\,\eta_{\mathrm{link}}\,\eta_{\mathrm{det}}
$$

再映射到每 bin 点击概率：

$$
p_{\mathrm{bg,bin}} = 1-\exp(-R_{\mathrm{bg}}\Delta t)
$$

该背景以“记忆模注入通道”进入状态端，不再走探测端 OR-map 临时补丁。

每步记忆模可注入的最小通道口径（单臂示意）：

$$
\Lambda_{\mathrm{bg}}(\rho_f)=\sum_\mu K_\mu\rho_f K_\mu^\dagger,
\quad
\sum_\mu K_\mu^\dagger K_\mu=I
$$

其中 $K_\mu$ 由 $p_{\mathrm{bg,bin}}$ 及偏振分配构造（例如 $H/V$ 对称或按实验不对称参数）。

---

## 4) 新方案程序流程 / 路径走向（闭环）

## 4.1 run 主流程

1. `trajectory.run_dual_atom_emission()`：仅负责 12D 发射，输出 `mps` 与发射诊断。
2. `common.run_emission_to_bs()`：在发射后插入 `QFC+滤波记忆` 状态端扫描阶段（新主路）。
3. `detection.run_detection_pipeline()`：
   - 构造检测端基础 effect（BS、探测效率、本征暗计数、$v_{res}$ 混合）；
   - 对状态端输出做记录概率收缩（不再构造 QFC/filter 记忆 MPO）；
   - 枚举 + 抽样 + 后验量子态输出。

## 4.2 记录概率算法（核心）

对每条候选记录（探测器组合 + bin 组合）：

1. 发射后，先在状态端执行一次 `QFC+滤波记忆` 扫描，得到 $|\Psi_{\mathrm{out}}^{\mathrm{aug}}\rangle$；
2. 从基础检测 effect 得到末端局域观测块 $\{E_n^{\mathrm{det}}\}$；
3. `core.DetectionContractionEngine` 用常规左/右环境收缩得到权重 $p_r$；
4. 归一化 $\{p_r\}$ 后进行 Monte Carlo 抽样。

## 4.3 复杂度口径（修订）

先明确：不同路线的复杂度不能只比较单个符号（如只比 $\chi^3$ 或只比 $81^2$），必须带上局域维度与收缩常数。

### A) 废弃路线：POVM-MPO 记忆主路

若把双臂记忆放到算符空间，按朴素口径会出现：

$$
O\!\left(B^2N\right),\quad B=9\times 9=81
$$

即 $81^2=6561$ 的键维乘子（再乘 MPS 收缩常数）。

### B) 执行路线：状态端显式记忆（本计划采用）

新增一段按 bin 扫描的 TEBD（每步两臂二体门 + 必要 swap），主开销口径：

$$
O\!\left(N\,C_{\text{2site}}(\chi,d_{\text{pair}})\right),
\quad d_{\text{pair}}=d_{\text{bin}}\times d_{\text{mem}}=5\times 3=15
$$

常用粗估可写成：

$$
C_{\text{2site}}\sim O\!\left(\chi^3 d_{\text{pair}}^3\right)
$$

但这只是上界口径，真实耗时强依赖实现常数、规范化频率、swap 调度和截断行为。

> 明确约束：**禁止**回到双臂 $B=81$ 的朴素 $O(B^2N)$ MPO 主路。

### C) 性能目标与止损阈值（避免“慢到不可用”）

1. 目标：新增 `QFC+滤波记忆` 阶段耗时控制在“单次发射 sweep”同量级的 $2\sim3\times$ 以内。  
2. 允许：在首版严格实现期，短期可到 $\le 5\times$，但需给出具体瓶颈分解。  
3. 止损：若稳定超过 $5\times$ 且无明确优化空间，暂停该实现并切换到“解析核入观测端”的备选方案（另立计划，不在本44内并行保留）。

工程策略：

1. 先实现**严格状态端记忆版**（不做近似裁剪）；
2. 以 profiling 驱动优化（gate 缓存、swap 路径优化、张量复用）；
3. 每一步优化必须保持同一物理语义，不引入兼容/回退分支。

## 4.4 基准与验收口径（新增）

固定基准命令（同随机种子、同参数）用于比较改造前后：

```bash
python total_simulation.py --runs 1 --shots 1 --debug
```

记录并对比以下字段：

1. `发射`
2. `QFC+滤波记忆`（新增）
3. `光纤`
4. `退相干`
5. `POVM构建`
6. `POVM枚举`
7. `POVM抽样`
8. `探测总计`
9. `核心阶段(去重)`
10. `run墙钟`

验收要点：

1. 新增阶段必须单独计时，禁止混在“探测总计”里。
2. 复杂度判断基于同命令实测，不接受拍脑袋分钟数。
3. 若性能/数学一致性冲突，优先保证数学一致性，并据实回报性能代价。

---

## 5) 修改后 tree（函数级，目标态）

> 下方是**实施完成后**的目标 tree（精确到关键函数）。

```text
total_simulation.py
├── _parse_run_params()
│   ├── (保留) --qfc-theta-h / --qfc-theta-v
│   ├── (新增) --qfc-phi-h / --qfc-phi-v
│   ├── (新增) --filter-fwhm-mhz
│   ├── (新增) --filter-eta-peak
│   ├── (新增) --filter-detuning-mhz-a / --filter-detuning-mhz-b
│   ├── (新增) --qfc-noise-sd-cps-per-mhz-a / --qfc-noise-sd-cps-per-mhz-b
│   └── (删除) QFC 背景旧口径参数（bg_rate_* 直传路径）
├── _build_task_list()
└── main()

atom_sim/experiment/common.py
├── class QfcParams
│   ├── theta_H / theta_V
│   ├── phi_H / phi_V
│   ├── filter_fwhm_mhz
│   ├── filter_eta_peak
│   ├── filter_detuning_mhz_A / filter_detuning_mhz_B
│   └── qfc_noise_sd_cps_per_mhz_A / qfc_noise_sd_cps_per_mhz_B
├── class NoiseParams
│   └── (保留) dark_rate_intrinsic_*（仅本征暗计数）
├── _build_run_parameter_store()
│   └── 改为谱密度口径生成 qfc 噪声概率
├── _build_detection_kwargs()
└── run_emission_to_bs()

atom_sim/physics/gates.py
├── qfc_gate(theta_H, theta_V, phi_H=0.0, phi_V=0.0)
├── build_detection_effects_6d()
├── build_detection_effects_9d()
├── build_arrival_projectors_5d()
├── filter_memory_step_unitary_3d()
├── build_filter_memory_transfer_one_arm()
├── build_filter_memory_transfer_pair()
├── build_qfc_filter_memory_step_ops()            # 新：状态端 QFC+记忆步进门/通道
├── apply_unitary_adjoint() / apply_local_channel_adjoint()
└── bs_gate_6d() / bs_gate_9d_dist()

atom_sim/core/mps.py
├── class DetectionContractionEngine
│   ├── from_mps()
│   ├── build_right_envs()
│   ├── build_left_envs()
│   ├── sum_same_bin()
│   └── sum_diff_bins()
└── compute_joint_arrival_probabilities()

atom_sim/simulation/detection.py
├── run_detection_pipeline()
│   ├── _build_bell_projector_full()
│   ├── _accumulate_success_and_fidelity()
│   └── (删除) _scale_qfc_source_background_map()
├── extract_qubit_state()
└── run_detection_self_checks()                   # 新增滤波记忆一致性条目

atom_sim/simulation/trajectory.py
├── run_dual_atom_emission()
├── apply_qfc_filter_memory_chain()               # 新主路：状态端记忆扫描
├── _prepare_filter_memory_sites()                # 新：记忆站点初始化/布局
└── _step_qfc_filter_memory()                     # 新：单 bin 步进（门 + Kraus + swap）

atom_sim/visualization/wavepacket.py
├── plot_dual_arm_heatmap()
├── _reconstruct_after_qfc_with_memory()          # 新：按记忆链重建
└── _reconstruct_after_fiber_with_memory()        # 新：按记忆链重建
```

---

## 6) 旧路径清理清单（必须删除）

1. 删除“QFC 背景 OR-map 旧路径”相关函数与调用。
2. 删除 `p_bg_qfc` 作为“直接探测端点击率”的旧语义。
3. 删除 `build_detection_effects_5d_by_bin()` 中 QFC/filter 的旧独立-bin实现；保留其纯探测端（fiber+BS+det）职责。
4. 删除 summary/readme 中旧字段描述，统一为新参数口径。
5. 删除任何“if new_path else old_path”分支。

---

## 7) 实施步骤（严格顺序）

1. **参数层改造**：`QfcParams` 与 CLI 全量替换（先删旧口径，再加新口径）。
2. **physics 层落地**：完成 `qfc_gate` 复相位与滤波记忆步进门/通道构造。
3. **simulation/trajectory 层落地**：插入 `apply_qfc_filter_memory_chain()` 主路径。
4. **simulation/detection 层切换**：删掉 QFC/filter 旧 effect 链，只保留探测端 effect。
5. **visualization 同步**：热图重建改为记忆链口径，删除独立-bin近似。
6. **文档/汇总收尾**：readme/tree/summary 字段同步，标注旧文档过时。

### 7.1 执行边界（当前阶段）

本轮仅允许修改计划文档与验收口径，不执行代码改动。

进入代码改造前必须满足：

1. 本 44 计划经人工确认；
2. 复杂度口径与止损阈值经人工确认；
3. 基准命令与统计字段经人工确认。

---

## 8) 验证标准（验收门槛）

1. `ruff check .` 通过。
2. `python total_simulation.py --self-check` 通过，且新增“滤波记忆一致性”子项。
3. `python total_simulation.py --runs 1 --shots 1 --debug` 可完整跑通并产出热图。
4. 关键行为校验：
   - 调小 `filter_fwhm_mhz` 时，时间相关增强（跨 bin 假符合结构变化可见）；
   - 关闭 QFC 噪声谱密度时，背景辅助成功率显著下降；
   - 调整 `phi_H/phi_V` 时，Bell 投影统计出现可解释偏移。

---

## 9) 风险与控制

1. **复杂度上升**：新增状态端 QFC+滤波 TEBD 扫描会显著增加耗时。  
   控制：先严格版，再做 gate 缓存与 swap 路径优化，不引入回退分支。
2. **可视化口径漂移**：热图若仍用旧重建会误导。  
   控制：与主算法同日切换，不留双实现。
3. **参数口径混乱**：新旧字段并存会制造技术债。  
   控制：旧字段直接删除，不做兼容。

---

## 10) 本计划的明确结论

1. 文档43给出的大方向可行，且与当前项目目标一致。  
2. 真正导致跨 bin 相关的是“滤波记忆”，不是 QFC 旋转本体。  
3. 本次 44 计划采用“**状态端显式滤波记忆模一次性替换**”路线，旧独立-bin路径全部清理。
