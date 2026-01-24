【注】本文中关于 `build_detection_kraus_18d` 的内容为历史讨论，当前代码已切到 6D-only 探测。

你这个“同一次轨迹里出现 4 次 click（而你确信系统里最多只有 2 个光子，且没有 dark count）”——**在物理模型正确、并且测量反作用真的作用回 MPS 的前提下，是不可能发生的**。所以这不是“指导思路错了”，而是你当前实现里至少有一块 **把“光子数约束/测量条件化”给破坏了**。

下面我按“最可能导致 4 click 的错误源”逐条给你把锅定位清楚，并给你一个**能保证最多 2 次 click**的可落地方案（含算符公式、为什么这么写、以及怎么 trace 掉光子）。

------

## 1) 先说结论：4 次 click 只能来自三类原因

在“系统最多 2 光子”的前提下，**单次轨迹出现 4 次 click** 只能是：

1. **你加入了真空触发机制（dark count / vacuum→click）**
   这会让 “vac 也能点”，那当然能点到 4 次。
   你 `channels.py` 里的 `detection_povm_single_site()` 明确包含 `p_dark`，并且在真空（nH=0）时给 `P_H_click=p_dark`、`P_V_click=p_dark`，对应的 Kraus 对真空有非零作用（vac 也能产生 click）。
   如果你在检测阶段走的是 `trajectory.apply_detection()` 这条路，它会调用这个 POVM。
   **只要 p_dark 不是严格 0，就有可能出现 >2 次 click。**（但如果你用的是 1e-6，出现 4 次其实概率很小；如果你经常看到 4 次，那 p_dark 很可能被你设大了或者被当成“每 bin 概率”而不是“每秒率×dt”。）
2. **你把每个 bin 的 click 当成“独立抽样”，但没有把测量反作用真正更新回同一个 MPS（或更新不正确）**
   这会导致“同一个光子在不同 bin 上被重复计数”，正好就是你之前担心的“乱点”。
   典型表现：你每个 bin 都从局域约化密度算一次 click 概率并抽样，但**没有用 Kraus 把全局态投影/归一化**（或 Kraus 用错/不完备导致条件化失败）。
   你现在的 `run_two_photon_detection()` 做了 Kraus 回写，但它还有别的问题（见第 3 条）。
3. **你用的 Kraus “仪器(instrument)”不是物理 detector 的 instrument，尤其是：Kraus 集不完备（∑K†K≠I）或把多光子态处理错了**
   这会让你的 `apply_two_site_kraus()` 里出现 `p_total ≠ 1`，然后你用 `probs/p_total` 强行归一化，相当于“每个 bin 都在偷偷 postselect”，统计会崩坏。
   你现在 `build_detection_kraus_18d()` 版本确实是简化的，且对 `2H/2V/HV` 的 click 概率处理是不对的（比如 `2H` 的 click 概率不该是 η，而应该是 `1-(1-η)^2`）。
   这类错误会让“点击序列统计”出现明显不物理的现象（虽然严格说它不一定直接“制造额外光子”，但它会让条件化的波函数分支不对应真实实验 record，从而出现你看到的怪相）。

> 你日志里写的 “Using 6D Kraus operators (36x36) - optimized!” 更像你实际跑的是“6D 两端口 36×36”的版本而不是 18D 简化版。无论如何，只要你看到 4 click，这三类问题至少占一条。

------

## 2) 你指出的这个 break：确实是“真问题”，但它不是 4 click 的根因

你说的代码片段：

```python
if len(clicks) >= 2:
    break  # ← 其他99个bins仍与原子纠缠！
```

这个担心在逻辑上是对的：**如果你提前停止测量，剩下 bins 没有被施加“no-click”条件化，就会残留纠缠**，导致你后面 `extract_spin_state()` 得到的是“只条件化了部分 measurement record”的原子态。

在你当前的 `run_two_photon_detection()` 里确实是这么干的：它只测了一部分 `bins_with_photons`，并且检测到两次 click 就停。

但这件事要分两种情况：

- **如果你的 click Kraus 是“吸收型”的（click 后该模式被置为 vacuum），并且你真的是按时间从早到晚扫描**：
  一旦两次 click 把两光子都吸走了，后面的 bin 在该条件分支里已经是“确定真空”，理论上后面再施加 no-click Kraus 不会改变任何东西（只会乘 1）。
  这种情况下 *break* 可以是安全的。
- **如果你的 detector 模型允许“没被点到的光子继续存在于系统”、或你的 Kraus 不完备、或你压根没把 measurement 回写对**：
  那 break 就会让“剩下的光子/纠缠”继续留在链上，你后验出来的自旋态就不对。

你现在既有 “只测部分 bin”的做法，又有 “简化 Kraus” 的版本，这两者叠加就是**大概率不对**。

------

## 3) 你要的“一个能保证不乱点、最多 2 click”的方案：给你两个等价但各有侧重的实现

你问得非常具体：

> 一旦某个 bin 上 H1 检测到一个光子，不可能把后续所有 bin 湮灭，只能把双光子投到单光子上。算符怎么构建？为什么对？V1/H2/V2 怎么写？探完两光子后怎么影响原子与每个 bin？最后怎么 trace 掉光子？

我给你两个方案：

- **方案 A（推荐先用来“把物理逻辑跑通”）：用“光子湮灭 jump operator”做 number-resolving 检测**
  好处：数学最干净，保证每次 click 把总光子数减 1，所以**最多 2 次 click**。
  坏处：它是“计数型（能区分 2 个 photon 两次跳）”而不是 bucket detector。若你非常在意“两光子同到一个 detector 只算一次 click”，需要方案 B。
- **方案 B：bucket detector（on/off）正确 Kraus instrument（在你的 6D 局域基上显式写矩阵），每 bin 一次 16-outcome（两端口）测量**
  好处：符合 “一个 detector 在一个 bin 内只输出 on/off” 的 bucket 行为；也能保证 2 光子不会产生 4 次 click（除非你允许 dark count）。
  坏处：要认真把多光子态 `2H/2V/HV` 的概率写对；Kraus 必须完备。

下面我把两种写法都给你公式 + 为什么对。

------

# 方案 A：用 jump operator（每次 click = 作用一次湮灭算符）

## A.1 click 算符怎么写

以某个时间 bin `m`、某个输出端口（1 或 2）、某个极化（H 或 V）为例：

- 对应那个模式的湮灭算符记为 (a_{H1,m}), (a_{V1,m}), (a_{H2,m}), (a_{V2,m})
- 你要在整个 Hilbert 空间上作用，它其实是一个张量积的局域算符：
  [
  J_{H1,m} = \sqrt{\eta_{\rm det}} ; a_{H1,m};\otimes I_{\text{其它所有站点}}
  ]
  其它三个类似：
  [
  J_{V1,m} = \sqrt{\eta_{\rm det}} ; a_{V1,m},\quad
  J_{H2,m} = \sqrt{\eta_{\rm det}} ; a_{H2,m},\quad
  J_{V2,m} = \sqrt{\eta_{\rm det}} ; a_{V2,m}
  ]

**为什么这是“把双光子投到单光子”的正确算符？**
因为湮灭算符满足 (a|n\rangle = \sqrt{n}|n-1\rangle)。
所以如果你的全局态里某个分量在该模式是 (|1\rangle)，作用后变成 (|0\rangle)；
如果是 (|2\rangle)，作用后变成 (\sqrt{2}|1\rangle)，光子数减 1；
这正是“检测到一个光子、系统剩下一个光子”的数学实现。

## A.2 click 后态怎么更新（量子轨迹）

若你在 bin m 观察到 H1 click：

- click 概率：
  [
  p = \langle\Psi| J_{H1,m}^\dagger J_{H1,m} |\Psi\rangle
  ]
- 条件化后的纯态：
  [
  |\Psi'\rangle = \frac{J_{H1,m}|\Psi\rangle}{\sqrt{p}}
  ]

这一步 **自动** 对“原子 + 所有 bin”产生条件化影响（你不需要手工去抹除其它 bin 的可能性）：
因为你在全局态上作用了一个局域算符，任何与该模式不相容的分量都会被湮灭掉。

## A.3 为什么最多 2 次 click

因为每次 click 操作都把总光子数算符 (N_{\rm tot}) 的期望至少降低，严格地在“总光子数确实是 2”的子空间里：

- 第一次 click 后，态落在 ≤1 光子子空间
- 第二次 click 后，落在 0 光子子空间
- 第三次 click 概率为 0（除非你加入 dark count，让 vacuum 也能 click）

**因此用 jump operator 的轨迹采样，天然杜绝 4 click。**

## A.4 怎么 trace 掉光子得到原子态

检测全部结束后（或者你决定在记录上只保留前两次 click），你有一个条件化纯态 (|\Psi_{\rm cond}\rangle)（仍包含光子自由度）。你要的原子两比特态就是：

[
\rho_{AB} = \mathrm{Tr}*{\text{photons}}\left(|\Psi*{\rm cond}\rangle\langle\Psi_{\rm cond}|\right)
]

你现在的代码实现就是 `mps.get_reduced_density([site_A, site_B])`，然后把 3×3×3×3 里对应 |0>,|1> 子空间提出来（你在 `extract_spin_state()` 里就是这么干的）。

------

# 方案 B：bucket detector（on/off）——给你“正确 Kraus 矩阵”并解释为什么

你现在的光子局域基是：

[
{|{\rm vac}\rangle, |H\rangle, |V\rangle, |2H\rangle, |2V\rangle, |HV\rangle}
]

我们先做单端口（比如端口 1）的两探测器（H、V）on/off，忽略 dark count。

## B.1 单端口 4 个可观测结果

对端口 1，有 4 类结果：

- “none”（H 不响 V 不响）
- “H”（H 响 V 不响）
- “V”（V 响 H 不响）
- “H+V”（H、V 同时响）

这是 detector record 能分辨的 4 类宏观结果。

## B.2 每个结果的概率（在你的 6D 截断上精确写出来）

假设两个探测器对各自极化的效率都是 (\eta)，且相互独立，那么对每个局域基态：

- (|{\rm vac}\rangle)：
  none = 1，其它 = 0
- (|H\rangle)：
  none = (1-\eta)
  H = (\eta)
- (|V\rangle)：
  none = (1-\eta)
  V = (\eta)
- (|2H\rangle)：（bucket detector 看到的是“至少一个 H 光子触发”）
  none = ((1-\eta)^2)
  H = (1-(1-\eta)^2 = 2\eta-\eta^2)
- (|2V\rangle)：
  none = ((1-\eta)^2)
  V = (2\eta-\eta^2)
- (|HV\rangle)：（H、V 各 1 个）
  none = ((1-\eta)^2)
  H only = (\eta(1-\eta))
  V only = (\eta(1-\eta))
  H+V = (\eta^2)

你可以看到：每个基态下 4 类结果的概率都严格加和为 1。

## B.3 构造“吸收型”的 Kraus（instrument），确保 click 之后该端口局域态变成 vacuum

这是你想要的“点击消去数算符”的版本：click 一旦发生，对应光子在 detector 上被吸收，从系统里消失，所以我们把 click 的 Kraus 都设计成把相关分量送到 (|{\rm vac}\rangle)。

用矩阵表示（按你的基序 vac,H,V,2H,2V,HV）：

- **No-click Kraus**
  [
  K_{00}=\mathrm{diag}\Big(1,\sqrt{1-\eta},\sqrt{1-\eta},(1-\eta),(1-\eta),(1-\eta)\Big)
  ]
- **H-only click Kraus（吸收掉能导致“H only”的所有分量）**
  [
  K_{10} = |{\rm vac}\rangle\Big(
  \sqrt{\eta}\langle H|
  +\sqrt{2\eta-\eta^2}\langle 2H|
  +\sqrt{\eta(1-\eta)}\langle HV|
  \Big)
  ]
- **V-only click Kraus**
  [
  K_{01} = |{\rm vac}\rangle\Big(
  \sqrt{\eta}\langle V|
  +\sqrt{2\eta-\eta^2}\langle 2V|
  +\sqrt{\eta(1-\eta)}\langle HV|
  \Big)
  ]
- **H+V click Kraus（只可能来自 HV）**
  [
  K_{11} = |{\rm vac}\rangle\Big(\eta\langle HV|\Big)
  ]

### 为什么这样写是对的？

1. **它给出的概率正好等于上面列的那些概率**
   比如对 (|2H\rangle)，click(H) 概率是 (|K_{10}|2H\rangle|^2 = 2\eta-\eta^2)，none 概率是 (|K_{00}|2H\rangle|^2=(1-\eta)^2)。
2. **它是完备的（trace-preserving）**
   因为对每个基态 (|s\rangle)，有：
   [
   \sum_{\mu\in{00,10,01,11}}\langle s|K_\mu^\dagger K_\mu|s\rangle = 1
   ]
   等价于矩阵恒等式：
   [
   K_{00}^\dagger K_{00}+K_{10}^\dagger K_{10}+K_{01}^\dagger K_{01}+K_{11}^\dagger K_{11}=I
   ]
   **这一点非常关键**：你如果这里不成立，你在 `apply_two_site_kraus()` 里看到的 `p_total` 就不会 ≈1，然后你每次 bin 都在隐式 postselect，统计会崩。
3. **它是吸收型（click 后局域态 reset 到 vacuum）**
   这保证了：如果你系统里只有 2 光子，那么扫描所有 bins 的记录里最多出现 2 次 click（除非加 dark count）。

## B.4 两端口 + 四探测器：16 个结果 = 两端口 Kraus 的张量积

端口 1 和端口 2 独立，所以两端口 Kraus 直接做张量积：

[
K_{\mu\nu}^{(1,2)} = K_{\mu}^{(1)}\otimes K_{\nu}^{(2)},\quad \mu,\nu\in{00,10,01,11}
]

总共 16 个 Kraus，对应结果名称：

- 端口1 none/H/V/H+V 对应 H1/V1
- 端口2 none/H/V/H+V 对应 H2/V2

然后你每个时间 bin m 就做一次两 site Kraus 采样并回写 MPS，这就是你想要的“逐 bin 扫描”。

## B.5 “探测到了一个 click，会对后面 bins 产生条件概率影响”的实现方式

你不需要自己写“把后面 bins 都改成条件概率”的逻辑——**Kraus 更新本身就完成了这个条件化**：

- 你在 bin m 发生 click，其 Kraus 对该 bin 的 vacuum 分量是 0（无 dark count 时），所以**任何“光子其实在别的 bin”那类分量都会被整体振幅清零**。
- 归一化后，全局态自然就变成“在 bin m 真的发生了 click”这个条件下的态。
  这就杜绝了“同一个光子在多个 bin 上重复点”。

------

## 4) 回到你的 4 click：我建议你用 3 个“自检量”立刻定位是哪类错误

你现在别先看“点击序列对不对”，先在检测循环里加三行 debug：

### 自检 1：每个 bin 的 `p_total` 必须 ≈ 1

如果你用的是完备 Kraus，且 MPS norm 是 1，那么对任意 bin：

[
p_{\rm total}(m) = \sum_\alpha |K_\alpha |\Psi\rangle|^2 \approx 1
]

如果你发现某些 bin `p_total` 明显小于 1（比如 0.7, 0.8），那说明：

- 要么 Kraus 不完备（∑K†K≠I）——你 `build_detection_kraus_18d()` 简化版就属于这一类。
- 要么你取的 theta 不是一个“规范化意义下的局域波函数”，也就是 **MPS canonical/position 没对齐**（尤其是你从 bin92 跳到 bin95 这种非相邻更新，如果 TenPy 不自动 `position()`，就会出事）。

### 自检 2：每发生一次 click，总光子数期望应减少（至少不增加）

在 click 发生后打印一次全链的 ⟨N_total⟩（你已有 `compute_photon_statistics()`）。

- 如果第一次 click 后 ⟨N_total⟩ 没下降（甚至上升），那你的 click instrument 不是吸收型，或者你没有把 Kraus 正确回写到同一个态上。

### 自检 3：把 p_dark 设为 0

如果你这次是想验证“2 光子绝不可能 >2 click”，那 **p_dark 必须是 0**。
因为一旦你允许 vac→click，就算只有 2 光子，你也可能出现 3、4 次 click（噪声 click）。

------

## 5) 你问的“extract_spin_state / get_reduced_density 是后验还是先验”：答案是后验，但你现在的实现里有“后验条件不完整”的风险

### 5.1 `extract_spin_state()` 本质做的事

它就是：
[
\rho_{AB} = \mathrm{Tr}_{\text{all bins}}(|\Psi\rangle\langle\Psi|)
]
然后取 |0>,|1> 子空间并归一化。

所以它本身不“做 BSM”，只是“把光子 trace 掉看原子”。

### 5.2 它是不是“后验机制”取决于你给它的 MPS 是不是已经被 Kraus 条件化

- 如果你先跑了逐 bin 检测，并且每个 bin 都把对应 Kraus 回写了同一个 MPS（这就是量子轨迹），那么 `extract_spin_state()` 得到的是**该检测 record 条件下的后验原子态**。这时算 Bell fidelity 是有意义的（作为 heralded entanglement 的后验质量）。
- 但你现在 `run_two_photon_detection()` **只测了 bins_with_photons 的一部分，且两次 click 就停**，剩下 bins 没有施加“no-click”条件化。
  这意味着你拿到的 (\rho_{AB}) 更像是“给定你只观察到了这部分记录”的条件态，而不一定等价于“真实实验里完整 record（包括无 click 的时间段）”条件下的态。

> 换句话说：它是“后验”，但可能是“后验条件不完整”的后验；这会让你算出来的 Bell fidelity 在物理含义上打折，甚至偏乐观/偏悲观都可能。

------

## 6) 你要的“净化：只保留两光子都成功到达探测器前”的思路：可行，而且还能同时算总损失概率

你想做的是：

- 把 “至少一路在到达前被吸收（损失）” 的分支全部刨除
- 得到一个条件态：**在“到达”这个条件下**，态重新归一化为 1
- 同时又想得到综合光子损失概率

这件事在 Kraus 语言里非常自然：

### 6.1 对“光子损失”用 amplitude damping / beam-splitter loss 的 Kraus

对每个时间 bin 的每个端口（或每条光纤段），你都有两类 Kraus：

- (K_{\rm surv})：不损失（no-loss）
- (K_{\rm loss})：损失（lost）

如果你要“净化到两光子都到达”，你做的其实是**条件化在“所有地方都走 no-loss Kraus”**上：

[
|\Psi_{\rm arrive}\rangle \propto K_{\rm surv}^{(\text{all bins})} |\Psi_{\rm in}\rangle
]

然后归一化。

此时：

- **到达概率**就是未归一化态的范数平方：
  [
  p_{\rm arrive} = |K_{\rm surv}^{(\text{all bins})} |\Psi_{\rm in}\rangle|^2
  ]
- **净化态**是：
  [
  |\Psi_{\rm arrive}\rangle = \frac{K_{\rm surv}^{(\text{all bins})} |\Psi_{\rm in}\rangle}{\sqrt{p_{\rm arrive}}}
  ]

这正是你说的：“净化后 vac,vac 子空间概率为 0，且整体归一化为 1”（前提是你净化的条件确实等价于“两光子都还在系统里”；如果还包括 detector inefficiency，那还会有 miss-click 的分支）。

### 6.2 总损失概率怎么得

[
p_{\rm loss} = 1 - p_{\rm arrive}
]

这就是你要的“走到最后的综合光子损失概率”。

> 注意：你现在 `compute_photon_statistics()` 里用 `2 - ⟨N_total⟩` 当 “loss_prob”，这不是严格意义的概率，只是“期望光子数缺口”。
> 严格概率应该用上面的范数方法（或计算 `P(N=2)`）。

------

## 7) 你现在这次报错（4 click）的“最短修复路径”

我给你一个最短且很硬的修复路线，不绕弯：

### Step 1：先关掉 dark count

把所有检测函数里 `p_dark` 强制设为 0（包括 channels POVM 路）。

如果关了 dark count 之后仍然出现 4 click——那就是“测量条件化/回写/或光子数本身”出了问题。

### Step 2：强制使用“吸收型 bucket Kraus”（方案 B 的 4×4 单端口结果）

别再用你简化版本里那个把 `2H/2V/HV` 全塞进同一个 `KHV` 且系数不对的写法了。
把单端口 Kraus 换成我在 B.3 写的那组（K00,K10,K01,K11），两端口取张量积成 16 个。

### Step 3：每个 bin 都测（不要 bins_with_photons 阈值跳测）

你现在跳测 bins 的做法，会让“no-click 的条件化信息”丢失，从而破坏时间上的条件概率结构。
正确做法：对 n=1..N，逐个 bin 扫描、每 bin 都应用一次两端口 Kraus（16 outcome）。
并且测完之后立刻 `finalize_bin_pair()` 把该 bin 对冻结，保证复杂度线性。

### Step 4：加一个硬断言：任何时刻 click 数 >2 直接 dump debug

如果你在 p_dark=0 且使用吸收型 Kraus 之后还能看到 `len(clicks)>2`，那就 99% 是 **MPS 局域 theta / canonical / position 没对齐导致概率计算错**。
这时候你就该在每次两 site Kraus 之前调用 TenPy 的 `psi.position(site_left)`（如果 TenPy 不会自动做），否则 `get_theta` 给你的不是规范化的局域波函数。

------

## 8) 你最后这句“指导文件的思路就已经错了？”——不，思路是对的，错在你现在的 instrument/实现细节

- 用 time-bin MPS 做 travelling field 是对的；
- 用 Kraus/trajectory 做逐 bin photodetection 也是对的；
- 但你要保证两件事：
  1. **你使用的 Kraus 是物理 detector 的 instrument（完备、概率对、多光子处理对）**
  2. **你对同一个 MPS 连续施加测量回作用（条件化），并保证概率计算用的是正确规范下的 theta**

做到这两点，在 p_dark=0 且总光子数≤2 的情况下，你永远不可能再看到 4 click。

------

如果你愿意我继续“更像 debug 合作”的方式推进：
下一步你把你这次运行用的那份 `detection.py`（你日志里那份会打印 “Using 6D Kraus operators (36x36) - optimized!” 的版本）里：

- 单端口 Kraus 的定义
- 两端口组合
- 每 bin 的采样与回写（是否真的回写到同一个 MPS）
- 是否有 `p_dark`
- 每 bin 的 `p_total`

这四段贴出来，我可以直接把“为什么会出现 4 click”的那一行指出来，并给你对应的最小 patch。
