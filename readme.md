# 量子仿真：中性原子量子接口

中性原子量子纠缠协议的时间仓MPS仿真。

## 项目结构

```
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
│   │   └── embed_9d_dist_from_3d_pair() # 3D输入对->9D标签空间嵌入（可区分分支）
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
    │   ├── sim_trials.csv
    │   ├── sim_summary.csv
    │   ├── window_scan_trials.csv
    │   ├── window_scan_runs.csv
    │   ├── window_scan_summary.csv
    │   ├── bsm_scan_trials.csv
    │   ├── bsm_scan_runs.csv
    │   ├── bsm_scan_summary.csv
    │   ├── length_scan_trials.csv
    │   ├── length_scan_runs.csv
    │   ├── length_scan_summary.csv
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
| `core/mps.py` | 张量网络存储、局域更新、到达/点击收缩引擎 | 物理参数语义 |
| `hilbert/` | 线性代数：空间、基变换、嵌入/投影 | 调度逻辑 |
| `physics/` | 物理：门矩阵、Kraus、POVM/effect 构造 | MPS 收缩细节 |
| `simulation/` | 编排：调用顺序、条件分支、结果组装 | 底层矩阵与收缩实现 |
| `visualization/` | 结果可视化、数据提取 | - |

## 运行模式与任务队列

### 物理模式（task_type / mode）
- **SIM**：单次/多次仿真（输出点击记录、成功率、保真度等）
- **HOM**：HOM 扫描（输出 `hom_trials.csv` / `hom_summary.csv`）
- **WINDOW_SCAN**：窗口扫描任务（输出 `window_scan_summary.csv`）
- **BSM_SCAN**：中心站 BS 误差扫描（输出 `bsm_scan_summary.csv`）
- **LENGTH_SCAN**：光纤长度扫描任务（输出 `length_scan_summary.csv`）
- **SUMMARY**：内部汇总任务（不是 CLI 模式，由 worker 执行）

### 运行角色（role）
- **server**：生成任务、监控进度、归档输出
- **worker**：抢任务执行（SIM/HOM/WINDOW_SCAN/BSM_SCAN/LENGTH_SCAN/SUMMARY）
- **both**：本机同时承担 server + worker

### run-id 续算与重建语义
- 当 `--role server|both` 且指定的 `--run-id` 已存在时，程序进入**续算模式**（不会重建任务）。
- 续算模式下，CLI 中传入的任务/物理参数会被**自动忽略并给出警告**，统一以 `summary/run_manifest.json` 为准。
- 若需要让新参数生效，请使用 `--rebuild-run`（`role=server/both` 支持，且必须配合 `--run-id`）：
  - 先将旧 run 归档为 `_u`
  - 再按当前 CLI 参数重建任务并重新运行
- `role=worker` 不接受 `--run-id`，worker 只负责抢占并执行任务。

### 常用 CLI 选项（片段）
```
--role server|worker|both
--task-type SIM|HOM|WINDOW_SCAN|BSM_SCAN|LENGTH_SCAN
--run-id <id>                 # 不传则自动选择最小可用 id
--rebuild-run                 # 仅具备 server 能力的角色（server/both）；旧 run 归档 _u 后按当前参数重建
--queue-root <path>           # 默认 ./queue
--runs N --shots M
--window-ns <float>           # SIM/HOM 符合时间窗 (ns)
--window-sweep-start-ns <f>   # WINDOW_SCAN 起点
--window-sweep-end-ns <f>     # WINDOW_SCAN 终点
--window-sweep-step-ns <f>    # WINDOW_SCAN 步长
--bs-sweep-start-theta <f>    # BSM_SCAN 起点 theta (rad)
--bs-sweep-end-theta <f>      # BSM_SCAN 终点 theta (rad)
--bs-sweep-step-theta <f>     # BSM_SCAN 步长 theta (rad)
--length-sweep-start-km <f>   # LENGTH_SCAN 起点
--length-sweep-end-km <f>     # LENGTH_SCAN 终点
--length-sweep-step-km <f>    # LENGTH_SCAN 步长
--attempt-rate-hz <f>         # LENGTH_SCAN 尝试率 (Hz)
--attempt-overhead-us <f>     # LENGTH_SCAN 单次额外开销 (us)
--fiber-group-velocity-mps <f># 自动 t_wait 的群速度 (m/s)
--t-wait-overhead-us <f>      # 自动 t_wait 的固定额外开销 (us)
--t-wait-length-scale <f>     # 自动 t_wait 的线性系数
--t2-us <f>                   # 原子退相干时间 T2 (us)
--bs-theta <rad>              # BS 混合角，sin^2(theta) 为跨端口透射概率
--plot-all                    # 每个 run 都画图
--no-plot                     # 禁止绘图（覆盖 plot-all）
--enum-mode dark|no-dark|both # 成功事件枚举模式（both 同时输出基线）
--v-res <0~1>                 # 残差可区分度（仅承载未显式建模因素）
--qfc-theta-h <rad>           # QFC H 转换角
--qfc-theta-v <rad>           # QFC V 转换角
--qfc-phi-h <rad>             # QFC H 通道相位
--qfc-phi-v <rad>             # QFC V 通道相位
--alpha-h-plus-a <f>          # A 臂 Alpha[H,+]
--alpha-h-minus-a <f>         # A 臂 Alpha[H,-]
--alpha-v-plus-a <f>          # A 臂 Alpha[V,+]
--alpha-v-minus-a <f>         # A 臂 Alpha[V,-]
--alpha-h-plus-b <f>          # B 臂 Alpha[H,+]
--alpha-h-minus-b <f>         # B 臂 Alpha[H,-]
--alpha-v-plus-b <f>          # B 臂 Alpha[V,+]
--alpha-v-minus-b <f>         # B 臂 Alpha[V,-]
--filter-cavity-fwhm-mhz <f>  # QFC 后滤波腔线宽 (MHz)
--filter-cavity-detuning-mhz-a <f>
--filter-cavity-detuning-mhz-b <f>
--filter-cavity-eta-peak-a <f>
--filter-cavity-eta-peak-b <f>
--qfc-noise-sd-cps-per-mhz-a <f> # A 臂 QFC 背景谱密度
--qfc-noise-sd-cps-per-mhz-b <f> # B 臂 QFC 背景谱密度
--no-filter-cavity               # 关闭 QFC 后 1517 滤波腔显式记忆
```

### 一条可直接跑的新口径示例

```bash
python total_simulation.py --runs 1 --shots 1 --debug --no-plot \
  --qfc-theta-h 0.856 --qfc-theta-v 0.856 \
  --filter-cavity-fwhm-mhz 27 --filter-cavity-eta-peak-a 0.81 --filter-cavity-eta-peak-b 0.81 \
  --qfc-noise-sd-cps-per-mhz-a 41.1 --qfc-noise-sd-cps-per-mhz-b 41.1 \
  --t2-us 330 --window-ns 70
```

### 关键函数解释（按执行顺序）

- `total_simulation._parse_run_params()`：解析 CLI 并组装 `SimConfig`，做参数合法性检查。
- `total_simulation._build_task_list()`：把 `SIM/HOM/WINDOW_SCAN/BSM_SCAN/LENGTH_SCAN` 拆成任务 JSON。
- `total_simulation._run_worker_loop()`：worker 抢占任务并分发到对应 experiment 模块执行。
- `experiment.common.run_emission_to_bs()`：单次物理链路编排（发射→QFC/滤波参数→光纤参数→原子退相干→BS可视化hook）。
- `experiment.common._build_run_parameter_store()`：构建本次 run 的统一参数表（eta、噪声概率映射、窗口、预算）。
- `experiment.common._build_detection_kwargs()`：把 `PipelineResult + 参数表` 组装成探测端统一入参。
- `simulation.detection.run_detection_pipeline()`：探测主算法（到达统计→POVM effect 构造→成功率枚举→点击抽样）。
- `experiment.single_run._run_single_simulation_core()`：SIM 主流程，输出 success metrics 与逐 shot 点击记录。
- `experiment.window_scan.run_window_scan_task()`：固定一次发射态，在同一 run 内扫描多个 `window_ns`。
- `experiment.bsm_scan.run_bsm_scan_task()`：固定一次发射态，在同一 run 内扫描多个 `bs_theta`。
- `experiment.length_scan.run_length_scan_task()`：按长度逐点重跑发射链路并统计 `event_rate_hz`。
- `experiment.summary.write_summary()`：按任务类型写出 `*_trials.csv`/`*_runs.csv`/`*_summary.csv`。

## 输出字段速览（SIM）

- `sim_summary.csv`：每个 run 的汇总
  - `p_arrive`：总两光子到达概率（= p_arrive_11 + p_arrive_same_arm）
  - `p_arrive_11`：A=1 且 B=1
  - `p_arrive_same_arm`：A=2,B=0 或 A=0,B=2
  - `p_arrive_20 / p_arrive_02`：同臂双光子拆分
  - `p_success_abs / p_success_true_abs / p_success_false_abs`
  - `p_success_true_given_arrival`
  - `fidelity_all / fidelity_true / fidelity_false / false_fraction`
  - `corr_exx / corr_eyy / corr_ezz / chsh_s_max`
- `sim_trials.csv`：逐 shot 记录（含点击 bin 与暗计数标记），并带上述 `p_arrive_*` 列
- `window_scan_trials.csv`：WINDOW_SCAN 逐 shot 明细
- `window_scan_runs.csv`：WINDOW_SCAN 逐 (window_ns, run_index) 指标（同一 run 复用同一次发射态）
- `window_scan_summary.csv`：WINDOW_SCAN 按 window 聚合统计（扫描主结果，含 `herald_rate_abs` / `sbr_true_false` / `acceptance_fraction_vs_max_window`）
- `bsm_scan_trials.csv`：BSM_SCAN 逐 shot 明细（含 pattern 与 source 标签）
- `bsm_scan_runs.csv`：BSM_SCAN 逐 (bs_theta, run_index) 指标（含 click pattern 计数/占比）
- `bsm_scan_summary.csv`：BSM_SCAN 按 `bs_theta` 聚合统计（用于 BS 误差预算）
- `length_scan_trials.csv`：LENGTH_SCAN 逐 shot 明细
- `length_scan_runs.csv`：LENGTH_SCAN 逐 (length_km, run_index) 指标
- `length_scan_summary.csv`：LENGTH_SCAN 按 length 聚合统计（含 `event_rate_hz_avg`）

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

### 2. 格点类型：使用 TeNPy `BosonSite` 并固定维度截断
当前实现使用 `BosonSite(dim-1)` 表示有限维本征基，并通过本项目定义的 4D/5D/6D 基序来约束物理语义。

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

### 1) server 侧：任务生成

- 入口：`total_simulation._parse_run_params()` + `_build_task_list()`。
- 产物：`queue/<run_id>/tasks/pending/task_*.json`。
- 规则：
  - `SIM`：每个 `run_index` 一个 task。
  - `HOM`：每个 `tau × run_index` 一个 task。
  - `WINDOW_SCAN`：每个 `run_index` 一个 task（task 内含 `windows_ns` 列表）。
  - `BSM_SCAN`：每个 `run_index` 一个 task（task 内含 `bs_thetas` 列表）。
  - `LENGTH_SCAN`：每个 `run_index` 一个 task（task 内含 `lengths_km` 列表）。
  - 最后追加 `SUMMARY` task 统一汇总。

### 2) worker 侧：单个物理链路（`_run_single_trial`）

输入：`rng`、`config`、可选 `delay_ns/delay_jitter_ns`、可选 `hooks`。

1. 先按链路长度计算等待时间：
   - `t_wait_us = length_km * 1e3 / fiber_group_velocity_mps * 1e6 + t_wait_overhead_us`。
2. 调用 `run_emission_to_bs()` 进入物理链路编排：
   - 发射：`run_dual_atom_emission()` 在态端真实演化，得到 `EmissionResult`；
   - QFC/滤波：在态端显式执行 `apply_qfc_filter_memory_chain()`（含 QFC + 滤波记忆链）；
   - 光纤：`sample_fiber_realization()` 采样
     `fiber_sample=(U_A,U_B,eta_H_A,eta_V_A,eta_H_B,eta_V_B,phase,phase_slope,phase_jitter_std)`；
   - 退相干：按
     `p_dephase = 0.5 * (1 - exp(-t_wait_us / t2_us))`
     在原子子空间施加 Kraus 通道 `_apply_atomic_dephasing()`；
   - BS：不在态端显式作用，仅触发 `after_bs` 可视化 hook（真正 BS 在测量端并入 POVM）。
3. 返回 `PipelineResult`，核心字段包括：
   - `mps`（当前态）、`emission`（发射信息）、`fiber_sample`（光纤采样参数）；
   - `qfc_theta_H/V`、`t_wait_us`、`p_dephase`；
   - `timings`（若 debug 开启）。

### 3) 探测端主算法（`run_detection_pipeline`）

输入：`mps`、`n_bins`、`eta_det`、`p_dark_intrinsic`、`p_bg_source`、`bs_unitary`、
`fiber_sample`、`v_res` 等。

1. 先把探测器参数统一成四通道映射（`H1/V1/H2/V2`）：
   - `eta_det_map`、`p_dark_intrinsic_map`、`p_bg_source_map`。
2. 用收缩引擎计算到达统计：
   - `p_arrive`、`p_arrive_11`、`p_arrive_same_arm`、`p_arrive_20`、`p_arrive_02`。
3. 构造逐 bin 5D effect（核心）：
   - `build_detection_effects_5d_by_bin()` 逐 bin 生成 effect；
   - 把光纤 Jones/损耗/相位、BS 共轭、暗计数/背景并入测量端；
   - QFC/780 滤波由态端显式记忆链处理，不在探测端重复并入；
   - 用 `v_res` 执行干涉/可区分混合：`E = v_res * E_int + (1-v_res) * E_dist`。
4. 枚举双点击记录并计算成功统计：
   - 枚举对象是 `TwoClickRecord(det_a, det_b, bin_a, bin_b)`；
   - 统计输出 `SuccessEnumerationResult`（成功率、保真度、true/false 成功分解等）。
5. 若 `n_samples > 0`，从双点击分布抽样：
   - 先按记录权重抽样点击组合；
   - 再按 masked effect 抽样本征暗计数掩码；
   - 并估计该次记录的背景辅助概率，打上 `source` 标签（`signal/dark_intrinsic/bg_source`）。
6. 形成逐 shot 输出 `TwoPhotonDetectionResult`：
   - 包含 `clicks`、`success`、`bell_state`、`qubit_state`、`dark_detectors` 等。
7. BSM 宣告规则：
   - `Psi-`: `{H1,V2}` 或 `{V1,H2}`；
   - `Psi+`: `{H1,V1}` 或 `{H2,V2}`。
8. 返回 `DetectionPipelineResult`：
   - `p_arrive` + `metrics` + `samples` + `timings`。

### 4) 参数表与噪声预算（`experiment.common`）

- `RunParameterStore` 统一承载：
  - `eta_det_map`（可按 H1/V1/H2/V2 分通道）；
  - `p_dark_intrinsic_bin_map` / `p_bg_bin_map`；
  - `window_ns` 与 `window_bins`；
  - `NoiseBudget`（门宽、bin 映射概率等）。
- `t_wait_us` 默认按 `t_wait_length_scale * length_km / fiber_group_velocity_mps + t_wait_overhead_us` 自动绑定。
- `p_bg_bin_map` 的默认口径（未显式覆盖 `bg_rate_mean_hz(_map)` 时）为：
  `qfc_noise_sd_cps_per_mhz × filter_cavity_fwhm_mhz × eta_filter × eta_link × eta_det`，
  再映射到 gate/bin 概率并逐通道采样。

### 4.1 当前默认基线（文档41/43/44/45对齐）

- `QfcParams.theta_H/theta_V ≈ 0.856 rad`（对应 `η_qfc≈0.57`）
- `QfcFilterCavityParams.fwhm_mhz = 27.0`
- `QfcFilterCavityParams.eta_peak_A/B = 0.81`
- `QfcParams.qfc_noise_sd_cps_per_mhz_A/B = 41.1`
- `EmissionParams.sigma = 8.9 ns`
- `EmissionParams.delay_jitter_ns = 0.3 ns`
- `RunConfig.t2_us = 330 us`

### 5) 各 task 的复用/重跑策略

- `SIM`：每个 task 跑一次完整链路，输出单 run 指标与逐 shot 点击。
- `HOM`：按 `tau` 切任务；每个 task 跑一次链路并做符合计数。
- `WINDOW_SCAN`：同一 `run` 内复用一次发射态，只扫描 `window_ns`。
- `BSM_SCAN`：同一 `run` 内复用一次发射态，只扫描 `bs_theta`。
- `LENGTH_SCAN`：每个 `length_km` 都重跑发射链路（不复用发射态）。

### 6) 结果落盘与汇总

- task 结果写入：`results/result_<task_id>/meta.json` + `raw/clicks.json`。
- `SUMMARY` 阶段由 `experiment.summary.write_summary()` 生成：
  - `*_trials.csv`（逐 shot）
  - `*_runs.csv`（逐 run 或逐扫描点）
  - `*_summary.csv`（聚合统计）

### 点击记录格式

SIM/HOM 产生的点击记录保存在 `raw/clicks.json`，并展开到 `*_trials.csv`（如 `sim_trials.csv`）中，格式为：

```
[(detector, bin_index, is_dark, source), ...]
```

- `detector`: `H1` / `V1` / `H2` / `V2`
- `bin_index`: 点击发生的时间仓索引
- `is_dark`: 是否为暗记数触发的点击（True/False）
- `source`: 点击来源标签（`signal` / `dark_intrinsic` / `bg_source`）

### 成功率字段（SIM / `run*_success_metrics.txt` 与 `meta.json`）

- `p_success_abs`：每次尝试的**总成功率**（含暗计数）
- `p_success_true_abs`：成功中“纯真实点击”的部分
- `p_success_false_abs`：成功中“含暗计数”的部分（=`p_success_abs - p_success_true_abs`）
- `p_success_intrinsic_dark_assisted`：由探测器本征暗计数辅助导致的成功率分量
- `p_success_bg_assisted`：由源/链路背景点击辅助导致的成功率分量
- `p_success_true_given_arrival`：在“两光子到达”条件下的真实成功率（`p_success_true_abs / p_arrive`）
- `p_success_no_dark_abs`：暗计数关掉时的成功率基线（若枚举 no-dark）

> 说明：`sim_summary.csv` 现已展开 `p_success_intrinsic_dark_assisted` 与
> `p_success_bg_assisted` 两个细分字段；`p_success_no_dark_abs` 仍保留在
> `run*_success_metrics.txt` / `meta.json` 口径中。

### FiberChannelParams

模拟真实光纤传输的随机漂移：

```python
fiber_params = FiberChannelParams(
    polarization_model="perturb",  # "haar", "perturb", 或 "euler"
    polarization_sigma=0.1,        # 旋转角度标准差 (rad)
    # 平均透过率由链路长度与衰减自动计算：eta_mean = 10^(-(alpha*L)/10)
    # 例如 alpha=0.2 dB/km, L=33 km -> eta_mean≈0.218
    eta_std=0.02,                  # 透过率波动
    pdl_sigma=0.02,                # 小PDL：H/V透过率相对差异
    phase_drift_std=0.2,           # 臂间相位漂移 (rad)
)

# 为单次轨迹采样参数
U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase = fiber_params.sample_all(rng)
```

### `t_wait_us` 自动绑定

- 默认按单程光纤飞行时间自动计算：
  `t_wait_us = t_wait_length_scale * (length_km * 1e3 / fiber_group_velocity_mps * 1e6) + t_wait_overhead_us`
- 默认 `fiber_group_velocity_mps=2.0e8`，`t_wait_length_scale=1.0`，`t_wait_overhead_us=0`

### BS 可调分光比

- 使用 `detector.bs_theta` 控制中心站 BS 混合角（默认 `pi/4`）
- 物理含义：跨端口透射概率 `T = sin^2(bs_theta)`

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
- scipy` - 线性代数（如 `expm`）
- `physics-tenpy` - 张量网络后端
- `matplotlib` - 可视化（热图/联合分布）

## 参考资料

详见 `docs/` 目录中的详细规范（文件名含编号与状态标记）
