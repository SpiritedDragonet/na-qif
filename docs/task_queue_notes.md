# 任务清单 + 抢占式节点方案（可执行版）

目标：在抢占式 HPC 节点随时可能回收的情况下，稳定完成长时间任务（>17h），并保证结果不会丢失。

本方案已拍板：**主节点磁盘作为共享目录（自建 NFS）+ 原子重命名领取任务 + Auto Scaling 自动补节点**。不再保留其他备选。

---

## 1. 固定决策与路径约定

固定决策：
- 使用主节点本地磁盘路径作为共享目录（主节点 NFS Server）
- NFS 为唯一任务队列与结果落地位置
- 任务领取靠“同文件系统内原子 rename”
- 节点自动化使用 Auto Scaling + user-data 启动 worker

路径约定（统一固定）：
```
主节点导出目录: /data/quantum_sim
工作节点挂载点: /mnt/quantum_sim
队列根目录: /mnt/quantum_sim/queue
```

目录结构（全部在 NFS 上）：
```
/mnt/quantum_sim/queue/
  tasks/
    pending/
    inprogress/
    done/
  results/
  summary/
  heartbeat/
```

---

## 1.1 total_simulation 架构改造（统一本地/远程）

目标：不区分“本地跑”和“远程跑”，统一为 **服务端生成任务 + 客户端 worker 领取任务**。

固定角色设计：
- `--role server`：只生成任务清单 + 汇总结果
- `--role worker`：只领任务 + 执行 + 写结果
- `--role both`：默认（本地跑时等价于“本地同时起 server 与 worker”）

固定队列参数：
- `--queue-root /mnt/quantum_sim/queue`：统一队列目录
- `--task-type HOM|SIM`：决定任务生成逻辑

自动化实例管理（固定选择）：
- **不由 Python 下单实例**，统一交给阿里云 Auto Scaling
- Python 只负责 server/worker 逻辑，不负责扩缩容

统一行为：
- 本地跑：`--role both`，server 生成任务，worker 领任务执行
- 远程跑：主节点 `--role server`，抢占式节点 `--role worker`

注：MQTT/RabbitMQ 不作为队列主通道，本方案固定使用共享目录；若需要节点上下线监控，仅通过 `heartbeat/` 文件判断。
注：server 角色应实现第 8 节的 Step 6/Step 7；worker 角色应实现第 3 节的领取/执行/完成流程。
命令模板（固定）：
```
# 主节点（服务端）
python total_simulation.py --role server --queue-root /mnt/quantum_sim/queue --task-type HOM --runs 120 --tau-start -40 --tau-end 40 --tau-step 2 --shots 1

# 抢占式节点（worker）
python total_simulation.py --role worker --queue-root /mnt/quantum_sim/queue --cores 64
```

---

## 2. 任务与结果文件格式（固定）

任务文件：`tasks/pending/task_{id}.json`  
结果目录：`results/result_{id}/`（可包含 JSON + PNG + CSV 等）

### 任务文件格式（JSON）
```
{
  "id": "hom_tau_-40_run_000123",
  "mode": "HOM",
  "tau_ns": -40.0,
  "shots": 1,
  "seed": 123456,
  "config_hash": "git:abcd1234"
}
```

### 结果目录结构（固定）
```
results/
  result_{id}/
    meta.json
    plots/
      *.png
    raw/
      *.csv
```

meta.json 示例：
```
{
  "id": "hom_tau_-40_run_000123",
  "status": "ok",
  "timestamp": "2026-01-30T12:34:56Z",
  "metrics": {
    "p_arrive": 0.0060,
    "coinc": 0,
    "valid": 1
  }
}
```

约束：
- `id` 唯一，全局不重复
- `config_hash` 固定写当前代码版本（例如 git commit hash）
- 结果目录结构固定如下（全部在 `results/result_{id}/` 下）：
  - `meta.json`：必须有，用于汇总
  - `plots/`：PNG 等图像输出（可选）
  - `raw/`：CSV 或二进制中间数据（可选）
- `meta.json` 写入采用 `.tmp -> rename`，避免半文件

---

## 3. Worker 领取/执行/完成流程（固定）

核心原则：**同一 NFS 内 rename 是原子操作**，天然互斥。

1) 领取任务（原子移动）
```
mv tasks/pending/task_X.json tasks/inprogress/task_X.json
```
成功即领取成功，失败则尝试下一个。

2) 执行任务
- 读取 task_X.json
- 执行 single_run
- 创建结果目录 `results/result_X/plots` 与 `results/result_X/raw`
- 写入 `results/result_X/meta.json.tmp`
- 原子重命名为 `results/result_X/meta.json`
- 其他输出写到 `results/result_X/plots/` 与 `results/result_X/raw/`

3) 完成标记
```
mv tasks/inprogress/task_X.json tasks/done/task_X.json
```
严格要求：**先写完结果，再移动到 done**。

4) 心跳（固定）
- 每 60 秒更新一次：
  - `heartbeat/worker_{hostname}.txt` 写入当前时间
- 同时每 60 秒 `touch tasks/inprogress/task_X.json`，续租约
5) 空任务退避（固定）
- 若当前无任务，worker 进入 sleep 退避（例如 5s -> 10s -> 30s），不退出
- 有新任务出现后自动继续领取

---

## 4. 租约与回收（固定规则）

规则：
- `tasks/inprogress` 里的任务以文件 `mtime` 作为租约时间
- 超过 10 分钟未更新视为失联，移回 `pending`
备注：worker 执行中必须定期 touch 该任务文件，否则长任务会被误判为失联。

回收流程（由主节点定时脚本执行，每 5 分钟）：
```
for each file in tasks/inprogress:
  if now - mtime > 10min:
     mv file -> tasks/pending/
```

---

## 5. 汇总流程（增量式固定）

规则：
- 主节点每发现一个 `results/result_{id}/meta.json` 就增量写入汇总表
- 以 `id` 去重，重复结果直接忽略

汇总文件路径：
```
/mnt/quantum_sim/queue/summary/hom_summary.csv
```

---

## 6. 节点自动化（固定）

采用 Auto Scaling 自动补齐抢占式节点：
- Desired capacity 固定为需要的节点数
- 节点被回收后自动拉起新节点

每个节点启动脚本（user-data）固定流程：
1) 挂载 NFS 到 `/mnt/quantum_sim`
2) 启动 worker（systemd 服务）
3) worker 在共享目录持续领取任务

---

## 7. worker 并发数 N（固定公式）

定义：
```
C = min(用户给的 cores, 机器实际核数)
reserve = 1
N = max(1, min(C - reserve, 待处理任务数))
```

说明：
- reserve 固定为 1，留给系统与日志
- worker 内部线程固定为 1（避免线程爆炸）

---

## 7.1 代码风格约束（固定）

 - 函数数量尽量少，避免过度拆分
 - 二层嵌套可用但数量不宜过多，三层嵌套禁止
 - 能用线性流程写清楚就不做额外抽象
 - 仅保留对整体架构必要的函数

---

## 8. 落地步骤（可执行清单）

### Step 1. 主节点配置 NFS Server
统一系统：Alibaba Cloud Linux。
在主节点执行：
```
sudo apt-get update
sudo apt-get install -y nfs-kernel-server
sudo mkdir -p /data/quantum_sim
sudo chown -R $USER:$USER /data/quantum_sim
```
配置导出（编辑 /etc/exports）：
```
/data/quantum_sim *(rw,sync,no_subtree_check,no_root_squash,fsid=0)
```
生效并验证：
```
sudo exportfs -ra
sudo exportfs -v
```
安全组/防火墙放行 NFSv4（TCP 2049）。

### Step 2. 在镜像/节点内配置 NFS 挂载
```
mkdir -p /mnt/quantum_sim
mount -t nfs -o vers=4.1,hard,timeo=600,retrans=2,noatime HEAD_NODE_IP:/ /mnt/quantum_sim
```
（写入 /etc/fstab，确保开机自动挂载）

### Step 3. 初始化队列目录
```
mkdir -p /mnt/quantum_sim/queue/tasks/pending
mkdir -p /mnt/quantum_sim/queue/tasks/inprogress
mkdir -p /mnt/quantum_sim/queue/tasks/done
mkdir -p /mnt/quantum_sim/queue/results
mkdir -p /mnt/quantum_sim/queue/summary
mkdir -p /mnt/quantum_sim/queue/heartbeat
```

### Step 4. 构建镜像（代码 + 依赖）
- 镜像内包含完整代码与 Python 依赖
- 代码路径固定为 `/opt/quantum_sim`
- Python 虚拟环境固定为 `/opt/venv`
- user-data 仅负责挂载 NFS 与启动 systemd 服务

systemd 服务名固定为：`quantum-worker.service`  
（服务内容随后在实现阶段补充，固定执行 `total_simulation.py --role worker`）
并设置：
- `Restart=always`
- `RestartSec=5`

### Step 5. Auto Scaling 配置
- Launch Template 指向该镜像
- Desired capacity = 需要的抢占式节点数
- 节点启动后自动运行 worker

### Step 6. 主节点生成任务清单
在 `tasks/pending/` 写入任务 JSON 文件（直接可执行示例）：
```
python - <<'PY'
import json
from pathlib import Path

ROOT = Path("/mnt/quantum_sim/queue")
PENDING = ROOT / "tasks" / "pending"
PENDING.mkdir(parents=True, exist_ok=True)

TAU_START = -40.0
TAU_END = 40.0
TAU_STEP = 2.0
RUNS = 120
SHOTS = 1
CONFIG_HASH = "git:abcd1234"

tau_values = []
v = TAU_START
while v <= TAU_END + 1e-9:
    tau_values.append(round(v, 6))
    v += TAU_STEP

task_id = 0
for tau in tau_values:
    for run in range(RUNS):
        task_id += 1
        tid = f"hom_tau_{tau:+.3f}_run_{run:06d}"
        task = {
            "id": tid,
            "mode": "HOM",
            "tau_ns": tau,
            "shots": SHOTS,
            "seed": 100000 + task_id,
            "config_hash": CONFIG_HASH,
        }
        with open(PENDING / f"task_{tid}.json", "w", encoding="utf-8") as f:
            json.dump(task, f, ensure_ascii=False)
print(f"已生成任务数: {task_id}")
PY
```

### Step 7. 主节点启动汇总进程
定时扫描 `results/` 并增量写入 `summary/hom_summary.csv`（直接可执行示例）：
```
python - <<'PY'
import csv
import json
import time
from pathlib import Path

ROOT = Path("/mnt/quantum_sim/queue")
RESULTS = ROOT / "results"
SUMMARY = ROOT / "summary" / "hom_summary.csv"
SEEN = ROOT / "summary" / "seen_ids.txt"

seen = set()
if SEEN.exists():
    seen.update(SEEN.read_text(encoding="utf-8").splitlines())

SUMMARY.parent.mkdir(parents=True, exist_ok=True)
if not SUMMARY.exists():
    with open(SUMMARY, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "p_arrive", "coinc", "valid", "timestamp"])

while True:
    for path in RESULTS.glob("result_*/meta.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tid = data.get("id")
        if not tid or tid in seen:
            continue
        m = data.get("metrics", {})
        with open(SUMMARY, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([tid, m.get("p_arrive"), m.get("coinc"), m.get("valid"), data.get("timestamp")])
        seen.add(tid)
    SEEN.write_text("\n".join(sorted(seen)), encoding="utf-8")
    time.sleep(10)
PY
```

---

## 9. 边界检查（必做）

- 任务 id 重复检测
- 结果写入成功再标记 done
- inprogress 租约回收生效
- 汇总去重生效
