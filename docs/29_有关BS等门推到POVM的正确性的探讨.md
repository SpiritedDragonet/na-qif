你这个顾虑非常正常：从“我把 BS / 光纤损耗这些器件都不再作用到态上”听起来确实像“把器件删了”。但如果你关心的是 **（i）每种点击记录的概率分布** 和 **（ii）按点击记录条件化后的原子（两原子）态**（你现在代码就是这么做的），那么把这些器件“推到 POVM effect 一侧”在数学上是**严格等价**的；它不是删器件，而是把**薛定谔绘景（动状态）\**换成\**海森堡绘景（动算符）**。这一点在量子信道/POVM 的标准形式里写得非常清楚。([Preskill Laboratory](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

下面我把它讲到“你能自己验证、也能对照你代码里的 P0 问题”的程度。

------

## 1）“推到 POVM 里”到底在做什么：不是删器件，是把“电路切口”往前挪

你现在的物理链路（抽象掉细节）可以写成：

- 初态（原子+光子） ：(\rho)
- 光学链路（BS、相位、偏振旋转、损耗、QFC+滤波……）看成一个整体量子信道：(\mathcal{E})
- 探测器给出一组 POVM：({E_r})，其中 (r) 就是你说的“bin + 四个探测器哪两个点了”的标签

那么**输出端**点击记录 (r) 的概率是标准 Born 规则：

[
p(r)=\mathrm{Tr}!\big[E_r,\mathcal{E}(\rho)\big].
]

海森堡绘景做的事是：把 (\mathcal{E}) 从“作用在态上”挪到“作用在算符上”，定义对偶（adjoint / dual）映射 (\mathcal{E}^*) 使得

# [ \mathrm{Tr}!\big[A,\mathcal{E}(\rho)\big]

\mathrm{Tr}!\big[\mathcal{E}^*(A),\rho\big].
]

在 Kraus 形式 (\mathcal{E}(\rho)=\sum_a M_a\rho M_a^\dagger) 下，

[
\mathcal{E}^*(A)=\sum_a M_a^\dagger A M_a.
]

这些等式和定义在John Preskill的讲义里是直接给出来的（他在“Quantum channels in the Heisenberg picture”一节把 (\mathcal{E}^*) 写得很明确）。([Preskill Laboratory](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

于是你可以定义“输入端等效 POVM”：

[
E_r^{\rm eff}=\mathcal{E}^*(E_r),
\quad\Rightarrow\quad
p(r)=\mathrm{Tr}!\big[E_r^{\rm eff},\rho\big].
]

**结论：BS/损耗/QFC/滤波都没有消失**，只是它们不再出现在“更新 MPS 的步骤”里，而是出现在“构造要收缩的测量算符”里。

------

## 2）你最担心的点：那原子后验态还对吗？

你现在在 `detection.py` 里做的事情，本质上是：给定一个点击记录 (r)，你用 effect（你代码里就是 `E_*`）去收缩光子自由度，得到一个**非归一化**的两原子密度矩阵 (\sigma_r)，最后再除以概率归一化。

关键事实是：

> **只要你最终把光子系统 trace 掉（你就是这么做的），那么对于只需要该分支的 effect (E_r=\sum_\alpha M_{r\alpha}^\dagger M_{r\alpha})，不需要知道每个 Kraus (M_{r\alpha}) 的细节。**

更形式化一点：若测量结果 (r) 的 CP 分支是 (\mathcal{J}*r(\cdot)=\sum*\alpha M_{r\alpha}(\cdot)M_{r\alpha}^\dagger)，则点击后原子系统（A）的非归一化态为

[
\sigma_r=\mathrm{Tr}_{\rm phot}!\big[(I\otimes \mathcal{J}_r)(\rho)\big].
]

而对任何原子可观测量 (A)，

# [ \mathrm{Tr}[A\sigma_r]

# \mathrm{Tr}!\big[(A\otimes \sum_\alpha M_{r\alpha}^\dagger M_{r\alpha}),\rho\big]

\mathrm{Tr}!\big[(A\otimes E_r)\rho\big].
]

这说明
[
\sigma_r=\mathrm{Tr}_{\rm phot}!\big[(I\otimes E_r)\rho\big].
]

所以当你把链路信道 (\mathcal{E}) 推到 effect 上，得到 (E_r^{\rm eff})，你的原子后验态也按同样方式变成

[
\sigma_r'=\mathrm{Tr}_{\rm phot}!\big[(I\otimes E_r^{\rm eff}),\rho\big],
]
并且它和“先对光子做 (\mathcal{E})，再用 (E_r) 收缩”的结果一致（本质还是上面那条对偶关系）。([Preskill Laboratory](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

**这正好解释了为什么你现在的 `detection.py` “effect-only 收缩”是自洽的，也解释了为什么 B 方案不会把原子后验态弄错。**

------

## 3）“BS 会被删掉吗？”——物理不删，实现上从“动状态”换成“动测量”

针对你最直观的担心，我用最具体的形式说：

- 若 BS 是 unitary (U_{\rm BS})，那么
  rm BS}^\dagger E^{\rm out} U_{\rm BS}.
  ]
  这就是“把 BS 并入 POVM”。([Preskill Laboratory](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))
- 你的代码里 BS 用 `bs_gate_6d()` 给出一个 36×36 的两模式门。
  方案 B 已将“BS 作用到 MPS”的路径彻底移除，改为在 `detection` 中为每个 outcome 构造 $E_r^{\rm eff}=U^\dagger E_r U$。

**一句话：测量算符的前置变换”。**

------

## 4）它和你们要修的 P0 是什么关系？——B 其实是“更干净地实现 P0”



> 不要再“固定选 K0（no-loss）后选”，而要把 QFC 失败 / 光纤损耗等真实地纳入统计，让成功率和 fidelity 随距离/噪声真实下降。

而你现在的代码里确实存在两个典型的“固定选 K0”的后选：

1. **780 滤波：**`apply_780_filter()` 固定用 `K_list[0]`（无 780 残留分支）并累积 `p_no_loss_780`，这等价于“只保留 QFC 成功（或 780 真空）的分支”。
2. **光纤损耗：**`apply_fiber_channel()` 固定用损耗 Kraus 的 `K_list_A[0], K_list_B[0]` 并累积 `p_no_loss_fiber`，这等价于“只保留全程无损耗的轨迹”。 样ion.run_detection_pipeline()`里算出来的`p_arrive / p_success`，本质是**在“已知一路都没丢光子”的条件下**的条件概率。 修是把大部分失败轨迹后选丢掉了”。

- 方案 A（全 Kraus 轨迹采样）能修，但会回到“成功极稀、需要海量 runs”的 Monte Carlo 地狱（你和同事已经意识到了）。
- 方案 B（把损耗/过滤推到 effec种确定性统计效率。

因此 **B 并没有否定 P0，而是用更高效、更不容易残留旧逻辑的方式实现 P0**。([Preskill Laboratory](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

------

## 5）B 方案“等价”的边界在哪里？什么时候你确实不该这么做？

非常关键，给你一个清晰的边界：

### B 方案严格等价的目标量

- 每个点击记录 (r) 的概率 (p(r))
- 以及你现在这种“收缩光子、只看原子”的条件化两原子态 (\sigma_r)（然后归一化）

在这些目标上，推到 effect 侧是严格等价的。([Preskill Laboratory](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

### B 方案不提供（或不适合）的东西

- **逐 bin 的量子跳跃轨迹细节**：比如“第 17 个 bin 先发生单击导致态先塌缩，然后第 23 个 bin 又发生什么”这种“时序塌缩过程”。那类东西本质上是“测量仪器（instrument）+ 轨迹”层面的，需要 MC/量子跃迁方法。

你现在代码生成双点击的方式，本来就是从一个（可能很大的）联合分布里直接抽样 outcome，而不是做逐 bin 的连续测量塌缩，所以 B 路线和你现有架构是匹配的。

------

## 6）你如何自己“100% 确认”它可行：两个最硬的 sanity check

如果你想要一个“让自己心里踏实、也能说服合作者”的验证方式，我建议你做两类 test（都能在很小维度上做，不需要跑整条链）：

### Test 1：BS 的等价性（不含损耗）

- 随机取一个两模（36D）纯态 (\rho)
- 用你现在的路径算：
  [
  p_{\rm out}(r)=\mathrm{Tr}[E_r,U_{\rm BS}\rho U_{\rm BS}^\dagger]
  ]
- 用 Heisenberg 算：
  [
  p_{\rU_{\rm BS}^\dagger E_r U_{\rm BS},\rho]
  ]
  两者必须数值一致（浮点误差量级）。([Preskill Laboratory](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

### Test 2：损耗信道的等价性（含 Kraus 和对偶）

- 随机取 (\rho) 和一个 effect (E)
- 显式算
  [
  p=\mathrm{Tr}[E,\sum_a M_a\rho M_a^\dagger]
  ]
- 对偶算
  [
  p=\mathrm{Tr}[(\sum_a M_a^\dagger E M_a)\rho]
  ]
  同样必须一致。([Preskill Laboratory](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

这两步一过，你就不会再把“缩进 POVM”理解成“删器件”了——它就是绘景选择。

------

## 7）对你现在工程结构的建议：你不需要“粗暴删器件”，可以做到可回退、可对照

你和同事聊的“删掉器件模型”的担心，其实可以用一种更工程化的方式化解：

- **保留** `bs_gate_6d()`、`loss_channel_*()` 这些“器件数学模型”作为公共库（它们仍然代表真实器件）。
- 但在“生产路径”里只保留一种计入方式：
  - 要么状态端 apply（薛定谔）
  - 要么 effect 端 backprop（海森堡）
  - **绝不能两边都做**（会双重计入）

为了避免“旧逻辑残留”，最干净的方法确实是：把 `apply_kraus_one_site_fixed(K0)` 这类后选路径从主线路移除/禁用，改为 effect 侧的 (\sum K^\dagger E K)。因为你现在的后选逻辑已经在 `trajectory.py` 明确写成“仅保留无损耗分支”。

------

### 最后一句话：你该怎么理解“B 路线”和 P0

- **P0 要的是“统计上把失败分支纳入”**（别再 K0 后选）。
- **B 路线提供的是一种在你当前式广系统），而是把混态效应折进测量算符，用对偶映射算出同样的概率与原子后验态。([Preskill Laboratory](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

如果你愿意，我也可以进一步把这句话“落地到你现有文件的最小改动策略”，但不需要你先答“要不要改”；我可以直接给你一份**“不重构大框架也能验证 B 的最小实验（one-bin sanity test）”**的步骤清单，让你先从“可验证”走到“可信任”。
