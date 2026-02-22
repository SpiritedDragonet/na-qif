# 量子仿真：中性原子量子接口

中性原子量子纠缠协议的时间仓MPS仿真。

## 项目结构

```
total_simulation.py                 # CLI 入口 + server/worker 调度
├── _parse_run_params()             # 解析并归一化 CLI 参数
├── _build_task_list()              # 生成 CORE_TRIAL 任务
├── _run_server_monitor()           # server 监控/ETA/stale 回收
├── _run_worker_loop()              # worker 抢占执行（含心跳）
└── main()                          # 角色分发与归档入口

atom_sim/
├── core/mps.py                     # MPS 状态与探测收缩引擎
├── hilbert/                        # 基空间与算符定义
├── physics/                        # 门矩阵、Kraus 通道、effect 构造
├── simulation/
│   ├── trajectory.py               # 发射/QFC/光纤/退相干链路编排
│   └── detection.py                # POVM 枚举、抽样、指标计算
├── experiment/
│   ├── common.py                   # SimConfig 与跨任务公共流程
│   ├── single_run.py               # SIM
│   ├── hom.py                      # HOM
│   ├── window_scan.py              # WINDOW_SCAN
│   ├── length_scan.py              # LENGTH_SCAN
│   ├── bsm_scan.py                 # BSM_SCAN
│   ├── qfc_noise_scan.py           # QFC_NOISE_SCAN
│   ├── detector_bg_scan.py         # DETECTOR_BG_SCAN
│   └── summary.py                  # SUMMARY 内部汇总任务
└── visualization/wavepacket.py     # 波包与阶段诊断图

queue/<run_id>/                     # server/worker 共享工作目录
├── tasks/
│   ├── pending/
│   ├── inprogress/
│   ├── done/
│   └── error/
├── results/
│   └── result_<task_id>/
│       ├── meta.json
│       ├── raw/                    # clicks.json / 调试文本 / 矩阵等
│       └── plots/
├── summary/
│   ├── run_manifest.json
│   └── *_runs.csv / *_summary.csv（部分模式额外产出 *_trials.csv）
└── heartbeat/
    ├── worker_<host>_<id>.txt
    ├── worker_output.txt           # 输出活跃标记（mtime）
    └── server_heartbeat.txt

outputs/<YYYYMMDD_HHMM>[_u]/        # run 归档目录
├── tasks/
├── results/
├── summary/
└── heartbeat/

docs/                               # 设计讨论与评审记录
thesis/                             # 论文源码与图片脚本
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
- **QFC_NOISE_SCAN**：QFC 背景噪声扫描（输出 `qfc_noise_scan_runs.csv` / `qfc_noise_scan_summary.csv`）
- **DETECTOR_BG_SCAN**：探测效率-背景二维扫描（输出 `detector_bg_scan_runs.csv` / `detector_bg_scan_summary.csv`）

`SUMMARY` 为内部调度模式（worker 在核心任务清空后自动执行），不是用户直接指定的物理模式。

### 运行角色（role）
- **server**：生成任务、监控进度、归档输出
- **worker**：抢任务执行（上述全部 `CORE_TRIAL` + 内部 `SUMMARY`）
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
--task-type SIM|HOM|WINDOW_SCAN|BSM_SCAN|LENGTH_SCAN|QFC_NOISE_SCAN|DETECTOR_BG_SCAN
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
--qfc-noise-sweep-start-cps-per-mhz <f>  # QFC_NOISE_SCAN 起点
--qfc-noise-sweep-end-cps-per-mhz <f>    # QFC_NOISE_SCAN 终点
--qfc-noise-sweep-step-cps-per-mhz <f>   # QFC_NOISE_SCAN 步长
--eta-det-sweep-start <f>      # DETECTOR_BG_SCAN 探测效率起点
--eta-det-sweep-end <f>        # DETECTOR_BG_SCAN 探测效率终点
--eta-det-sweep-step <f>       # DETECTOR_BG_SCAN 探测效率步长
--bg-mean-sweep-start-hz <f>   # DETECTOR_BG_SCAN 背景均值起点
--bg-mean-sweep-end-hz <f>     # DETECTOR_BG_SCAN 背景均值终点
--bg-mean-sweep-step-hz <f>    # DETECTOR_BG_SCAN 背景均值步长
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
- `total_simulation._build_task_list()`：把 `SIM/HOM/WINDOW_SCAN/BSM_SCAN/LENGTH_SCAN/QFC_NOISE_SCAN/DETECTOR_BG_SCAN` 拆成 `CORE_TRIAL` 任务 JSON。
- `total_simulation._run_worker_loop()`：worker 抢占任务并分发到对应 experiment 模块执行。
- `experiment.common.run_emission_to_bs()`：单次物理链路编排（发射→QFC/滤波参数→光纤参数→原子退相干→BS可视化hook）。
- `experiment.common._build_run_parameter_store()`：构建本次 run 的统一参数表（eta、噪声概率映射、窗口、预算）。
- `experiment.common._build_detection_kwargs()`：把 `PipelineResult + 参数表` 组装成探测端统一入参。
- `simulation.detection.run_detection_pipeline()`：探测主算法（到达统计→POVM effect 构造→成功率枚举→点击抽样）。
- `experiment.single_run._run_single_simulation_core()`：SIM 主流程，输出 success metrics 与逐 shot 点击记录。
- `experiment.window_scan.run_window_scan_task()`：每个 `run_index` 跑一次链路并输出窗口扫描基础记录，窗口聚合在 summary 端完成。
- `experiment.bsm_scan.run_bsm_scan_task()`：每个 `(bs_theta, run_index)` 跑一次链路并输出单点结果。
- `experiment.length_scan.run_length_scan_task()`：按长度逐点重跑发射链路并统计 `event_rate_hz`。
- `experiment.qfc_noise_scan.run_qfc_noise_scan_task()`：按 QFC 噪声谱密度逐点重跑并输出扫描结果。
- `experiment.detector_bg_scan.run_detector_bg_scan_task()`：按 `eta_det × bg_mean_hz` 网格逐点重跑并输出扫描结果。
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
  - `fidelity_true_avg` 采用全局条件口径：`sum(fidelity_true_run * p_success_true_abs_run) / sum(p_success_true_abs_run)`（非 run 等权平均）
- `bsm_scan_trials.csv`：BSM_SCAN 逐 shot 明细（含 pattern 与 source 标签）
- `bsm_scan_runs.csv`：BSM_SCAN 逐 (bs_theta, run_index) 指标（含 click pattern 计数/占比）
- `bsm_scan_summary.csv`：BSM_SCAN 按 `bs_theta` 聚合统计（用于 BS 误差预算）
- `length_scan_trials.csv`：LENGTH_SCAN 逐 shot 明细
- `length_scan_runs.csv`：LENGTH_SCAN 逐 (length_km, run_index) 指标
- `length_scan_summary.csv`：LENGTH_SCAN 按 length 聚合统计（含 `event_rate_hz_avg`）
- `qfc_noise_scan_runs.csv`：QFC_NOISE_SCAN 逐 (qfc_noise_sd_cps_per_mhz, run_index) 指标
- `qfc_noise_scan_summary.csv`：QFC_NOISE_SCAN 按噪声谱密度聚合
- `detector_bg_scan_runs.csv`：DETECTOR_BG_SCAN 逐 (`eta_det`, `bg_rate_mean_hz`, run_index) 指标
- `detector_bg_scan_summary.csv`：DETECTOR_BG_SCAN 二维聚合统计

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
  - `WINDOW_SCAN`：每个 `run_index` 一个 task（窗口聚合在 `summary` 阶段做）。
  - `BSM_SCAN`：每个 `bs_theta × run_index` 一个 task。
  - `LENGTH_SCAN`：每个 `length_km × run_index` 一个 task。
  - `QFC_NOISE_SCAN`：每个 `qfc_noise × run_index` 一个 task。
  - `DETECTOR_BG_SCAN`：每个 `(eta_det, bg_mean_hz) × run_index` 一个 task。
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
- `EmissionParams.sigma = 9.5 ns`
- `EmissionParams.t0_A_ns/t0_B_ns = 8.0 ns`（延迟与抖动按 `delay_ns/delay_jitter_ns` 在运行时拆分到 A/B）
- `EmissionParams.arm_A/B.omega_peak = 0.65 * (2*pi*20e6) rad/s`
- `EmissionParams.arm_A/B.g = 0.09 * (2*pi*20e6) rad/s`
- `EmissionParams.arm_A/B.kappa_ex / kappa_in = 20e6 / 1e6 (1/s)`
- `EmissionParams.delay_jitter_ns = 0.3 ns`
- `RunConfig.t2_us = 330 us`

### 5) 各 task 的复用/重跑策略

- `SIM`：每个 task 跑一次完整链路，输出单 run 指标与逐 shot 点击。
- `HOM`：按 `tau` 切任务；每个 task 跑一次链路并做符合计数。
- `WINDOW_SCAN`：每个 `run_index` 跑一次链路，窗口维度在汇总阶段统一重算。
- `BSM_SCAN`：每个 `(bs_theta, run_index)` 独立重跑链路。
- `LENGTH_SCAN`：每个 `length_km` 都重跑发射链路（不复用发射态）。
- `QFC_NOISE_SCAN`：每个 `qfc_noise_sd_cps_per_mhz` 都重跑链路。
- `DETECTOR_BG_SCAN`：每个 `(eta_det, bg_rate_mean_hz)` 都重跑链路。

### 6) 结果落盘与汇总

- task 结果写入：`results/result_<task_id>/meta.json` + `raw/clicks.json`。
- `SUMMARY` 阶段由 `experiment.summary.write_summary()` 生成：
  - `*_trials.csv`（逐 shot）
  - `*_runs.csv`（逐 run 或逐扫描点）
  - `*_summary.csv`（聚合统计）
  - `QFC_NOISE_SCAN` 与 `DETECTOR_BG_SCAN` 当前输出 `*_runs.csv` + `*_summary.csv`（无 `*_trials.csv`）

### 点击记录格式

点击记录保存在各 task 独占的 `raw/clicks.json`，并在汇总阶段展开到 `*_trials.csv`（如 `sim_trials.csv`）中，格式为：

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
- 默认 `fiber_group_velocity_mps=2.0e8`，`t_wait_length_scale=2.0`，`t_wait_overhead_us=0`

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

## 吞吐异常排查（线程风暴）

当出现“`inprogress` 上升但 `done` 长时间不增长”时，先检查是否发生线程风暴。

- `systemctl status quantum-worker` 中的 `Tasks` 是 **进程+线程总数**，不是队列 `task_*.json` 数量。
- 若 `Tasks` 高到数千，通常是每个 worker 进程都拉起了大量 BLAS/OpenMP 线程，导致严重过订阅。
- 当前项目已在 `total_simulation.py` 的数值库导入前设置：
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`

建议排查命令：

```bash
# 1) 服务健康与重启次数
systemctl status quantum-worker -n 40 --no-pager -l
systemctl show quantum-worker -p MainPID -p NRestarts -p ActiveState -p SubState

# 2) 每个 worker 子进程线程数（观察是否异常高）
MAIN=$(systemctl show -p MainPID --value quantum-worker)
for p in $(pgrep -P "$MAIN" python3.11); do
  printf "%s " "$p"
  awk '/Threads/{print $2}' /proc/$p/status
done | sort -k2 -n | tail -n 20

# 3) 心跳与任务触碰是否仍在更新
find /root/quantum_project/queue/<run_id>/heartbeat -name 'worker_*.txt' -mmin -2 | wc -l
find /root/quantum_project/queue/<run_id>/tasks/inprogress -name 'task_*.json' -mmin -3 | wc -l
find /root/quantum_project/queue/<run_id>/tasks/inprogress -name 'task_*.json' ! -mmin -3 | wc -l
```

补充：
- `heartbeat/worker_output.txt` 仅用于“近期有无输出”的 mtime 标记，不是完整日志文件。
- worker 机器上的真实错误信息请看 `journalctl -u quantum-worker`。

## 依赖项

- `numpy` - 数组操作
- `scipy` - 线性代数（如 `expm`）
- `physics-tenpy` - 张量网络后端
- `matplotlib` - 可视化（热图/联合分布）

## 参考资料

详见 `docs/` 目录中的详细规范（文件名含编号与状态标记）
