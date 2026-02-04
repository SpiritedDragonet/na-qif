你现在这套 **5D bin + 把 QFC/过滤/光纤/BS 全推到 POVM（Heisenberg 对偶）** 的逻辑，**在“每臂每个 bin 最多 1 个光子”这一物理截断成立的前提下，是可以闭环的**；而且你写的 **36 → 9 → 25** 这条维度链，**方向是对的**，并且“看起来够又不够”的那种直觉，正好对应：

- **输出端（BS 后）确实需要 6D/36D 才能容纳 bunching 的 2 光子态**；
- **但输入端（BS 前）确实只需要 3D/9D（每臂 0/1 光子）**；
- **再往前（QFC/过滤之前）才需要把 780 的“失败分支”记账进来，所以是 5D/25D**。

下面我按你要求的“编译器式 type-check”把维度链条完整写出来，并且把当前代码里涉及的所有门/通道逐一标维度、标它们在 Heisenberg 端怎么乘进 POVM。

------

## 1) 先把三套 Hilbert 空间说清楚（这是闭环的关键）

### (A) 你想要的 5D（态端 bin）

每一臂每个 bin：
[
\mathcal H_{5}=\mathrm{span}{ |0\rangle, |H_{780}\rangle, |V_{780}\rangle, |H_{1517}\rangle, |V_{1517}\rangle }
]
所以 **一对臂（A,B）同一个 bin 的联合维度**：
[
\mathcal H_{5}^{A}\otimes \mathcal H_{5}^{B} ;\Rightarrow; 5\times 5 = 25
]

> 这正是你要的“保留 QFC 失败分支（仍在 780）但不引入 18D 直积”的最小账本维度。

### (B) BS 前的 3D（只剩 1517 的 0/1 光子子空间）

BS 之前（过滤后、光纤中）**物理上只有 1517**，且每臂每 bin 仍最多 1 个光子：
[
\mathcal H_{3}=\mathrm{span}{ |0\rangle, |H\rangle, |V\rangle}
]
两臂同 bin 联合维度：
[
\mathcal H_{3}^{A}\otimes \mathcal H_{3}^{B};\Rightarrow;3\times3=9
]

### (C) BS 后/探测端的 6D（必须能描述 2 光子 bunching）

探测端每个输出端口（记为 port1/port2）需要：
[
\mathcal H_{6}=\mathrm{span}{ |0\rangle, |H\rangle, |V\rangle, |2H\rangle, |2V\rangle, |HV\rangle}
]
两端口联合维度：
[
\mathcal H_{6}^{(1)}\otimes \mathcal H_{6}^{(2)};\Rightarrow;6\times6=36
]

你的 `build_detection_effects_6d()` 正是在这个 36D 上构造 POVM element（effect）矩阵的。

------

## 2) 为什么 “36 → 9 → 25” 是对的（不是“删掉器件”，是换边算）

你想要的等价恒等式就是：
[
\mathrm{Tr}[E_{\text{out}},\Phi(\rho)] ;=; \mathrm{Tr}[\Phi^{\dagger}(E_{\text{out}}),\rho]
]
其中 (\Phi) 是“从发射后到探测器前”的整条量子通道（含 BS、损耗、QFC、过滤等），(\Phi^\dagger) 是其对偶（Heisenberg 端）。这就是“把门写进 POVM”的严格数学意义。

**关键点：维度变化发生在“通道连接处”，你必须用投影/嵌入矩阵把空间对齐**。所以你这条链的每一步，其实都对应一个“空间接口”：

- **36D**：探测 POVM 的自然空间（BS 后输出端口，需含 2 光子）
- **9D**：BS 输入的可达空间（每臂最多 1 光子）
- **25D**：更前端（QFC 前）为了“记住失败分支”的最小账本空间（含 780 单光子）

------

## 3) 你要的“编译器式”完整算式与维度对齐

下面我严格按你“每一步矩阵维度必须对得上”的要求写。

### 3.1 先定义两个接口矩阵（你文档里叫 P/Π）

#### (i) 6D → 3D 的投影（其实是把 effect 限制到输入可达子空间）

选取 ({|0\rangle,|H\rangle,|V\rangle}) 对应 6D 基序的前三个：
[
P_{3\leftarrow 6}\in \mathbb C^{3\times 6}
]
它满足 (P_{3\leftarrow6}P_{3\leftarrow6}^{\dagger}=I_3)，而 (P_{3\leftarrow6}^{\dagger}P_{3\leftarrow6}) 是投影到 6D 的单光子子空间。

两端口：
[
\Pi_{3\leftarrow6}=P_{3\leftarrow6}\otimes P_{3\leftarrow6}\in \mathbb C^{9\times 36}
]

#### (ii) 3D → 5D 的嵌入（把 telecom 子空间塞进 5D 的 (vac, H1517, V1517) 槽位）

你的 5D 基序是 ((0, H_{780},V_{780},H_{1517},V_{1517}))，因此
[
P_{5\leftarrow 3}\in \mathbb C^{5\times 3}
]
把 3D 的 ((|0\rangle,|H\rangle,|V\rangle)) 映射到 5D 的索引 ((0,3,4))。

两臂：
[
\Pi_{5\leftarrow3}=P_{5\leftarrow3}\otimes P_{5\leftarrow3}\in \mathbb C^{25\times 9}
]

------

### 3.2 现在把所有“门/通道”按物理顺序推回去（每步标维度）

**起点：探测 POVM effect（36×36）**
`build_detection_effects_6d()` 生成的每个 outcome 的 effect：
[
E_{\text{out}} \in \mathbb C^{36\times 36}
]

------

#### Step 1：把 BS 推到 effect 上（仍在 36D）

BS 门是 `bs_gate_6d()`，它是 **36×36** 的两端口酉门。

[
E_{\text{bs}} = U_{\text{BS}}^{\dagger},E_{\text{out}},U_{\text{BS}}
\qquad (36\times36)
]

> 注意：这里 BS 没“消失”，只是从“作用在态上”变成“共轭作用在 effect 上”。

------

#### Step 2：限制到 BS 输入可达子空间（36 → 9）

因为你坚持 5D bin（每臂每 bin 最多 1 个光子），所以 BS 输入端每个端口最多 1 光子 ⇒ 6D 的双光子基矢永远不会被输入态占据。

因此可把 effect 限制到输入子空间：
[
E_{3}=\Pi_{3\leftarrow6},E_{\text{bs}},\Pi_{3\leftarrow6}^{\dagger}
]
维度检查：
[
(9\times36),(36\times36),(36\times9)=9\times9
]

这一步就是你写的 **36 → 9**。

------

#### Step 3：把光纤（Jones/相位/损耗）推到 effect 上（9D 不变）

你当前光纤在态端做了：

- `jones_gate()` 6×6 单站点酉门（含单光子与双光子子空间的 (U\otimes U) 作用）
- `loss_channel_1517_raw()` 给出 6×6 的损耗 Kraus 列表
- 然后在 `apply_fiber_channel()` 里 **固定取 K_list[0] 当 no-loss 后选**，并且还叠加了 bin 相关的相位 profile（phase_slope/jitter）

在 5D 方案里你要的是：**这些不再改态，而是进入 (\Lambda_{\text{fiber}}^\dagger)**。

因为我们现在已经在 3D（vac,H,V）子空间，所以你完全可以用 3D 版本的 fiber channel（更小更干净）：

- Jones：(U_{\text{J},3}=\mathrm{diag}(1,U_{2\times2})\in\mathbb C^{3\times3})
- 损耗（偏振依赖）：3 个 Kraus（0/1 光子幅度阻尼）即可，输出仍在 3D

两臂（A/B）独立时：
[
\Lambda_{\text{fiber}} = \Lambda_A \otimes \Lambda_B
]
对偶：
[
E_f = \Lambda_{\text{fiber}}^{\dagger}(E_{3})
\qquad (9\times9 \rightarrow 9\times9)
]

------

#### Step 4：嵌入回 5D 账本空间（9 → 25）

现在我们要把 “telecom 的 effect” 变成 “作用在 5D 账本空间上的 effect”，好让后面可以吃进 **780 分支（QFC 失败）**。

[
E_{5}=\Pi_{5\leftarrow3},E_f,\Pi_{5\leftarrow3}^{\dagger}
]
维度检查：
[
(25\times9),(9\times9),(9\times25)=25\times25
]

这就是你写的 **9 → 25**。

------

#### Step 5：780 过滤（以及你想保留的“失败→真空→暗计数假成功”）进入对偶映射（25D）

你现在的 `apply_780_filter()` 是：构造 `loss_channel_both_subspaces()`（18D Kraus），然后只取 `K_list[0]` 并 `apply_kraus_one_site_fixed()` 做后选。
这正是 P0 要修的点：**把失败分支扔掉了**。

在 5D 中，过滤可以写成 5×5 Kraus（你文档里那三个 K 就是一个正确的 TP 集）：

- (K_0)：保留 vac 与 1517 单光子，杀掉 780 分量
- (K_1=|0\rangle\langle H_{780}|)
- (K_2=|0\rangle\langle V_{780}|)

两臂：
[
\Lambda_{\text{fil}} = \Lambda_{\text{fil},A}\otimes \Lambda_{\text{fil},B}
]
对偶：
[
E_{\text{fil}}=\Lambda_{\text{fil}}^{\dagger}(E_{5})
\qquad (25\times25 \rightarrow 25\times25)
]

------

#### Step 6：QFC 进入对偶（25D）

你现在的 `qfc_gate()` 是在 18D（780×1517）上构造 18×18 酉门。
在 5D 版本里它会是 5×5：分别在 ((H_{780},H_{1517})) 和 ((V_{780},V_{1517})) 两个 2×2 子空间做旋转，vac 不变。

两臂：
[
U_{\text{qfc},AB}=U_{\text{qfc}}\otimes U_{\text{qfc}}\in \mathbb C^{25\times25}
]
对偶（酉门的对偶就是共轭）：
[
E_{\text{in}} = U_{\text{qfc},AB}^{\dagger},E_{\text{fil}},U_{\text{qfc},AB}
\qquad (25\times25)
]

------

### 3.3 最终闭环：这 25×25 就能“正确作用于你的 5D MPS”

对任意一个点击 outcome (r)（你代码里用 key 表示 detectors 列表），最终你插入 MPS 收缩用的就是：
[
E^{(r)}*{\text{eff}} = E*{\text{in}}^{(r)} \in \mathbb C^{25\times25}
]
然后概率权重就是 Born rule：
[
p(r)=\mathrm{Tr}\big[E^{(r)}*{\text{eff}};\rho*{\text{emission}}\big]
]
你现在 detection pipeline 做的“枚举/抽样双点击记录”，本质上就是对这些 (p(r)) 做归一化抽样。

------

## 4) 结论：36→9→25 **对路**，而且“够不够”取决于你是否接受两个物理前提

### 前提 1：每臂每个 bin 最多 1 个光子（单光子近似）

你目前模型里 780 子空间本来就是 3D（vac,H,V），从源头上禁止了同一臂同一 bin 的双光子。
所以 **在你当前代码物理假设下**，BS 输入端确实只需 3D，不需要 6D。

如果你未来要包含“同一臂同一 bin 的双发射/多光子”，那 5D 会不够，你就必须把输入端也扩到能含 2 光子的 6D（甚至更大），那时就不是 36→9→25 了，而是 36→36→… 或 36→(≥36)→…（因为输入端也可能占据双光子子空间）。

### 前提 2：BS/光纤不跨 bin（无色散/无时间混叠）

你现在的 BS 是逐 bin 作用在 (A_n,B_n) 上的 `apply_bs()`。
光纤相位 profile 也是逐 bin 单站点门。
在这个假设下，“把它们推到 POVM”仍然是逐 bin 的（只是变成逐 bin 的 effect 变换）。

但如果你真的要研究色散（跨 bin 卷积），那你说得对：
**MPS 的复杂度不会消失，只会从“态端多体门”迁移到“测量端 MPO（跨 bin 的 effect）”。**
这时你要的就不是一个 25×25 的局域 effect，而是一个跨多个 bin 的 MPO。

------

## 5) 你担心的那句：“都缩进 POVM 后，会不会分不清真纠缠 vs 暗计数假成功？”

**单次实验记录当然分不清**——真实实验也分不清。
但仿真里你仍然能做两件关键的、对实验很有指导意义的事：

1. **宣告成功的后验原子态**（这就是你论文里要报的 fidelity）
   你用 (E_{\text{eff}}^{(r)}) 去收缩 MPS 得到原子 reduced state，这里暗计数/损耗分支已经“混进来了”，所以它给的是**实验可比的“宣告成功态”**。
2. **“真成功/假成功”分解的统计**
   你当前 `build_detection_effects_6d()` 已经同时返回了 `effects_all` 和 `effects_true`：

- `effects_true` 是“没有 dark_detectors 的那部分 Kraus”累加出来的 effect
- `effects_all` 是把 dark 也包含后的总 effect

你在 `single_run.py` 里输出的 `p_success_true / p_success_false` 也是基于这套拆分做的。

把损耗/QFC 失败推到 POVM 后，这种拆分反而更“物理正确”，因为你终于把
[
P(\text{success}|\text{loss})
]
那一项（靠暗计数误判的假成功）真正统计进去了，而不是被后选删掉。

------

## 6) 你问的“BS 推到 POVM 后，BS 这个门到底是多少维？”

**BS 仍然是 36×36**，一点没变：`bs_gate_6d()` 就是 36×36。
变化只在于：你不再把它 `apply_bond_op()` 到 MPS 上，而是对 effect 做：
[
E \leftarrow U_{\text{BS}}^{\dagger} E U_{\text{BS}}
]
然后再用 (\Pi_{3\leftarrow6}) 把它压回输入子空间变成 9×9（再嵌入 25×25）。

换句话说：

- **门的维度没有变**（还是 36×36）
- **只是你把它“乘在 effect 那边”**，并且利用输入态的可达性把它压缩成 9×9 有效 POVM。

------

## 7) 给你一句“总判断”，回答你最后那句直觉

> “36 → 9 → 25 看起来够又不够”

**正确理解是：它就是“够且刚好够”，因为它分别对应三个不同位置的最小必要空间。**

- **36**：为了“输出端口可能出现 2 光子 bunching + 桶式探测”的正确 POVM（必须）
- **9**：为了“BS 输入端永远不占据双光子子空间”的可达约束（能压缩就压缩）
- **25**：为了“QFC 失败仍在 780，需要保留到过滤/暗计数统计里”的账本（必须）

只要你接受你现在模型的单光子截断，那么这条链不仅对路，而且是非常干净、可解释、也更容易写进论文的方法（你可以把它说成“在输入单光子子空间上构造等效 POVM”）。

------

如果你下一步还想让我继续“像编译器一样”检查你文档里每一条（比如你表格里把 BS 写成“单端口 6×6”这类表述是否会误导、或者 fiber 的 6D→3D 投影该怎么严格定义），我可以按你文档的每一行把“数学对象类型/维度/是否真的在现有代码里存在”逐条核对一遍，并指出哪里需要改表述或改实现入口（特别是 bin 相关 phase profile 会让 effect 变成按 bin index 不同的对象，这在 detection 的环境收缩里是一个真实的结构性影响）。