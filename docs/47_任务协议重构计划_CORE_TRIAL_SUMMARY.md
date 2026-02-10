# 47_任务协议重构计划_CORE_TRIAL_SUMMARY



重构目标：

1. 把“执行语义”和“实验语义”解耦，避免当前 `task.mode` 既当执行分发键又当实验类型键造成混乱。
2. 所有子任务尽量统一为“单次核心试验”语义（`CORE_TRIAL`），只保留一个 `SUMMARY` 汇总任务。
3. 子任务必须自描述，不依赖 worker 启动 CLI 的隐式参数。
4. 保持快速迭代原则：不做旧协议兼容、回退、降级路径。

---

## 1. 现状问题（从代码路径归纳）

当前任务生成在 `total_simulation.py::_build_task_list()`，worker 执行分发在 `total_simulation.py::_run_worker_loop()`。

当前痛点：

1. `mode` 语义混杂
   - 现状：`mode` 直接等于 `SIM/HOM/WINDOW_SCAN/BSM_SCAN/LENGTH_SCAN/SUMMARY`。
   - 问题：`mode` 同时承担“执行器分发”和“实验定义”，职责耦合。

2. 配置来源不封闭
   - `WINDOW_SCAN` task 当前不携带 `window_sweep_*`，但执行器又在 worker 侧校验这些配置必须存在。
   - 在分布式（server 与 worker 启动参数不同步）时会触发高并发秒失败。

3. 子任务粒度不统一
   - `SIM/WINDOW_SCAN/HOM` 大多是“每 task 一次核心试验”。
   - `BSM_SCAN/LENGTH_SCAN` 是“每 task 内部多点循环”。
   - 这导致 task 时间尺度与失败恢复颗粒度不一致。

4. Summary 与任务耦合不清
   - 一部分扫描逻辑在 task 内，一部分在 summary 内，关注点未彻底分离。

---

## 2. 新协议总览

### 2.1 顶层语义分层

新协议只允许两种执行模式：

- `mode = "CORE_TRIAL"`
- `mode = "SUMMARY"`

实验语义单独放在字段：

- `experiment in {"SIM", "HOM", "WINDOW_SCAN", "BSM_SCAN", "LENGTH_SCAN"}`

### 2.2 配置分层

将参数拆成两层：

1. **run 级配置（全局）**：写入 `queue/<run_id>/summary/run_manifest.json`
2. **task 级参数（局部）**：写在每个 task 的 `payload`

worker 执行时必须先加载 manifest，再读取 task payload；不再依赖 worker CLI 隐式注入。

---

## 3. 数据模型设计

## 3.1 run_manifest（全局配置）

建议结构：

```json
{
  "protocol_version": "v2_core_trial",
  "config_hash": "git:...",
  "task_type": "WINDOW_SCAN",
  "config": { "...": "SimConfig 全量快照" },
  "scan": {
    "window_sweep_start_ns": 0.0,
    "window_sweep_end_ns": 49.0,
    "window_sweep_step_ns": 1.0,
    "tau_values_ns": null,
    "bs_theta_values": null,
    "length_values_km": null
  }
}
```

说明：

- `config` 放完整 `SimConfig` 快照，确保 worker 与 server 一致。
- `scan` 放实验扫描轴定义，供 task 生成和 summary 聚合统一使用。

## 3.2 CORE_TRIAL task

通用字段：

```json
{
  "id": "core_trial_000123",
  "mode": "CORE_TRIAL",
  "experiment": "WINDOW_SCAN",
  "run_index": 123,
  "shots": 1,
  "seed": 100124,
  "config_hash": "git:...",
  "payload": {}
}
```

`payload` 按 experiment 细分：

- `SIM`: `{}`
- `WINDOW_SCAN`: `{}`  
  （窗口扫描在 summary 做，CORE_TRIAL 只产出无窗口限制点击记录）
- `HOM`: `{"tau_ns": <float>}`
- `BSM_SCAN`: `{"bs_theta": <float>}`
- `LENGTH_SCAN`: `{"length_km": <float>}`

## 3.3 SUMMARY task

```json
{
  "id": "summary",
  "mode": "SUMMARY",
  "summary_for": "WINDOW_SCAN",
  "config_hash": "git:..."
}
```

---

## 4. 参数归属矩阵（重点）

目标：明确“哪些参数给 CORE_TRIAL 单跑，哪些参数给 SUMMARY 聚合”。

### 4.1 CORE_TRIAL 使用参数

来自 manifest 的全局配置：

1. 发射链路参数：`emission.*`
2. QFC/滤波参数：`qfc.*`
3. 光纤参数：`fiber.*`
4. 探测参数：`detector.*`
5. 噪声参数：`noise.*`
6. 运行控制参数（影响单次物理）：`run.t2_us`、`run.window_ns` 等

来自 payload 的局部参数：

1. `HOM`：`tau_ns`
2. `BSM_SCAN`：`bs_theta`
3. `LENGTH_SCAN`：`length_km`
4. `SIM/WINDOW_SCAN`：无局部参数

### 4.2 SUMMARY 使用参数

来自 manifest 的扫描定义：

1. `WINDOW_SCAN`：`window_sweep_start_ns/end_ns/step_ns`
2. `HOM`：`tau_values_ns`（或根据 manifest 规则重建）
3. `BSM_SCAN`：`bs_theta_values`
4. `LENGTH_SCAN`：`length_values_km`

来自结果目录：

1. 每个 CORE_TRIAL 的 `meta.json`
2. 每个 CORE_TRIAL 的 `raw/clicks.json`

注意：summary 不应读取 worker CLI 参数。

---

## 5. 各实验的子任务切分规范

统一目标：每个 CORE_TRIAL 尽量只做一次核心物理链路，保证调度与恢复粒度一致。

### 5.1 SIM

- 任务数：`runs`
- 每 task：一次核心链路 + `shots` 次抽样
- summary：输出 `sim_summary.csv`

### 5.2 WINDOW_SCAN

- 任务数：`runs`
- 每 task：一次核心链路 + `shots` 次抽样（`window_bins=None` 保存无窗点击）
- summary：按 `window_sweep_*` 重放窗口判定，输出窗口曲线

### 5.3 HOM

- 任务数：`runs × len(tau_values)`
- 每 task：`payload.tau_ns` 指定延迟
- summary：按 `tau_ns` 聚合符合率与相关统计

### 5.4 BSM_SCAN

- 任务数：`runs × len(bs_theta_values)`
- 每 task：`payload.bs_theta` 单点扫描
- summary：按 `bs_theta` 聚合统计

### 5.5 LENGTH_SCAN

- 任务数：`runs × len(length_values_km)`
- 每 task：`payload.length_km` 单点扫描
- summary：按长度聚合统计与速率

---

## 6. 执行流设计（worker）

worker 执行顺序固定：

1. 读取 task JSON
2. 若 `mode == CORE_TRIAL`：
   - 读取 `run_manifest.json`
   - 校验 `config_hash`
   - 构造 trial 参数（manifest + payload）
   - 调统一核心入口：`run_trial_detection_core(...)`
3. 若 `mode == SUMMARY`：
   - 读取 `run_manifest.json`
   - 调对应 summary writer

分发条件只看 `mode`，实验分支看 `experiment`。

---

## 7. 错误处理与可观测性

1. schema 校验失败应写成结构化错误：
   - `error_type = "SCHEMA_ERROR"`
   - `error_detail = "missing payload.bs_theta"`

2. config 漂移错误：
   - `error_type = "CONFIG_MISMATCH"`
   - `error_detail = "config_hash mismatch"`

3. 任务执行错误：
   - `error_type = "RUNTIME_ERROR"`
   - `error_detail = exception message`

4. 避免 silent fallback：缺字段必须报错，不自动使用默认值掩盖问题。

---

## 8. 改造步骤（实施计划）

### 阶段 A：协议落地（先改队列协议，不改物理算法）

1. 新增 manifest 写入与读取
2. 将 task 统一改为 `CORE_TRIAL/SUMMARY`
3. 引入 `experiment + payload` 字段
4. worker 按新协议分发

### 阶段 B：各实验切分统一

1. 将 `BSM_SCAN/LENGTH_SCAN` 从“task 内多点循环”改为“每 task 单点”
2. summary 改为按单点任务聚合

### 阶段 C：清理旧路径

1. 删除旧 `mode=WINDOW_SCAN/HOM/...` 分支
2. 删除对旧 task 结构的兼容判断
3. 更新 README 任务协议章节

---

## 9. 验收标准

必须全部满足才算完成：

1. `SIM/HOM/WINDOW_SCAN/BSM_SCAN/LENGTH_SCAN` 五类任务都能在 `server+worker` 分离部署下稳定运行。
2. worker 启动时不传任何实验参数，仅传 `--role worker --queue-root ...` 也能正确执行。
3. `WINDOW_SCAN` 不再出现 `window_sweep_*` 缺失错误。
4. 任意 task 文件单独拷贝到另一台 worker 仍能由 manifest + payload 完整复现语义。
5. 代码中不再存在旧 `mode` 作为实验语义的分发路径。

---

## 10. 本计划的关键取舍

1. 统一单点 task 会增加 task 总数，但换来更强的可恢复性与可观测性。
2. 不做旧协议兼容，直接切到新协议，避免继续积累技术债。
3. summary 只做聚合，不再承担物理配置兜底。

以上即“没刀计划”基线。确认后按此计划开改。
