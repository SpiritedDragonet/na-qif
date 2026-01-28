【注】本文中关于 `build_detection_kraus_18d` 的内容为历史讨论，当前代码已切到 6D-only 探测。

你这个输出里出现 **4 次 click**（而且分布在 4 个不同 bin），在你自己设定的前提下——**系统总光子数最多 2、没有暗计数、探测是破坏性的**——确实是“不可能事件”。所以结论先放在最前面：

**不是指导思路错了，而是你的“探测/点击”实现没有真正把全局态按测量记录做条件塌缩（或 click Kraus 没有真正湮灭光子），等价于把同一粒子在不同 bin 上重复计数，才会出现 4 click。**

下面我按“你现在的代码结构”把问题定位得更具体，并给你一个能直接落地的方案（包括算符公式、为什么这样对、以及怎么作用在 MPS 上）。

------

## 1) 先对照你上传的 `detection.py`：**按这份代码跑，不可能得到 4 click**

你上传的 `run_two_photon_detection()` 逻辑是：

- 逐 bin 做两端口联合 POVM（Kraus）采样；
- 每次采样后调用 `mps_work.apply_two_site_kraus()` 把选中的 Kraus 真正作用到 **全局 MPS** 并归一化；
- 一旦 `clicks` 里累计到 2 个事件就 `break`。

所以**如果你真的跑的是这份 `detection.py`**，它最多只会打印到两次 click，然后就停了，不可能继续打印 bin 98、99。
你贴的日志里还出现了 **“Using 6D Kraus operators (36x36) - optimized!”** 这句，而你上传的 `run_two_photon_detection()` 并不会打印这句话。这两点结合起来，基本可以断定：

> 你本地实际跑到的探测函数不是这份 `run_two_photon_detection()` 的版本（要么 import 走错模块、要么你本地文件和你上传的不一致、要么你走了别的探测路径）。

这一步不是“抬杠”，而是直接解释了为什么你能看到 4 个 click：**你跑的那条路径很可能还是“每个 bin 独立抽一次 click”的老逻辑**（不对全局态做条件更新），这正是会把同一光子在多个 bin 上重复抽到 click 的典型症状。

------

## 2) “为什么会重复 click？”——核心物理点：**一次 click 必须让全局态条件化**

正确的量子测量更新永远是：

$$
\rho ;\to; \rho_\mu
=\frac{K_\mu,\rho,K_\mu^\dagger}{\mathrm{Tr}(K_\mu\rho K_\mu^\dagger)}
$$

- $K_\mu$ 只作用在“当前被测的那一个 bin（或该 bin 的两端口）”上；
- 但因为你的光子-原子-其余 bins 是纠缠的，所以局域 Kraus 会把整个波函数都条件化（这就是你想要的“点一下后，后面 bins 的条件概率改变”）。

如果你只是对每个 bin 计算一个概率然后独立掷骰子，而 **不把选中的 Kraus 真正作用回 MPS**，你得到的就不是“同一次实验的测量记录”，而是“把同一个波包在不同 bin 上重复抽样”。这样出现 4 次 click 就很自然了（本质上你抽的是“边缘分布”，不是“条件分布链”）。

你现在的 `MPSState.apply_two_site_kraus`$以及别名 `apply_two_site_kraus`$做的就是上面的条件化：它会对每个 Kraus 算 $p_\mu = |K_\mu\theta|^2$，按概率采样 $\mu$，然后把 $K_\mu$ 作用到该 bin 的两-site 张量上并归一化。
所以**只要你确实在每个 bin 都调用它更新态**，在“最多两光子、无暗计数、click 会吸收光子”的前提下，click 总数绝不可能超过 2。

------

## 3) 你问的“点一下如何把双光子投到单光子？”——最直接的算符公式

### 3.1 数分辨 / 量子跳跃（最干净、最不容易出 bug 的版本）

把 BS 后的 4 个探测模式记为 $\alpha\in{H1,V1,H2,V2}$，时间 bin 为 (n)。令该模式的湮灭算符为 $d_{\alpha,n}$。

“在 $(\alpha,n)$ 发生一次光子计数(click)”的标准 jump 更新就是：

$$
J_{\alpha,n}=\sqrt{\eta_\alpha},d_{\alpha,n},
\qquad
|\Psi\rangle\to \frac{J_{\alpha,n}|\Psi\rangle}{\sqrt{\langle\Psi|J_{\alpha,n}^\dagger J_{\alpha,n}|\Psi\rangle}}.
$$

- $d_{\alpha,n}$ 会把该模式的光子数减 1：$|2\rangle \to \sqrt{2}|1\rangle$，所以它天然实现你说的“**双光子投到单光子**”。
- 它不会“把后续所有 bin 湮灭”，只在当前 bin 的当前模式做局域更新；后续 bins 的变化来自纠缠条件化。

如果你只想先保证“不会出现 4 click 这种物理不可能”，这套 jump 形式是最稳的（因为每次 click 明确减少总光子数）。

### 3.2 SNSPD 桶式(on/off)版本（更贴近你说的“一个 bin 一个探测机会”）

桶式探测器的 POVM（单模式）满足：

- no-click 的概率：$(1-\eta)^n$（n 个光子都没触发）
- click 的概率：$1-(1-\eta)^n$

在你截断的 1517 子空间基
${|vac\rangle,|H\rangle,|V\rangle,|2H\rangle,|2V\rangle,|HV\rangle}$
且一个端口内有 H/V 两个独立桶式探测器时，四种端口结果（00/10/01/11）的**基态概率**就是：

$$
P_{00}=(1-\eta)^{n_H}(1-\eta)^{n_V}
$$
$$
P_{10}=\big[1-(1-\eta)^{n_H}\big](https://chatgpt.com/c/1-\eta)^{n_V}
$$
$$
P_{01}=(1-\eta)^{n_H}\big[1-(1-\eta)^{n_V}\big]
$$
$$
P_{11}=\big[1-(1-\eta)^{n_H}\big]\big[1-(1-\eta)^{n_V}\big]
$$

然后你把它做成 Kraus，就能保证：

- 真空不会 click；
- $|2H\rangle$ 的 click 概率是 $1-(1-\eta)^2=2\eta-\eta^2$，而不是 $\eta$；
- **不会把 $|2H\rangle$ 错算成 “H+V 同时 click”**（这是你现在实现里最危险的简化点之一，见下面第 4 节）。

------

## 4) 你现有探测 Kraus 里确实有一个“会搞乱结果”的地方：多光子部分太简化

你上传的 `run_two_photon_detection()` 用的是 `build_detection_kraus_18d()`。讨论文件已经点得很明白：
你目前最大的风险不在 `extract_spin_state()`，而在 **探测 POVM/Kraus 的多光子部分是否物理一致**。

尤其是这条物理约束：

> 对 n=2 的福克态，桶式 click 概率必须是 $1-(1-\eta)^2$，并且 “H+V 同时 click” **不该由 $|2H\rangle$ 或 $|2V\rangle$ 贡献**。

如果你把 $|2H\rangle$ 也塞进 “H+V” 结果里，然后又把 “H+V” 拆成两个 click 事件（H 和 V），那你会把 **“两光子 bunching 到同一偏振同一路”** 的情况人为变成 **“同一路两探测器同时响”**，这会把 click 统计、BSM 成功判据都搞偏。

这虽然不一定直接导致“跨不同 bin 的 4 click”，但会让你的模式分布整体不物理，间接制造很多怪现象。

------

## 5) 给你一个“不会乱点”的可执行点击算法（逐 bin + 条件化 + 最多两次 click）

下面是我建议你最终采用的“逻辑骨架”，它满足你要的两点：

- click 不会把后续 bins 直接清空，但会正确条件化后续 bins（因为更新了全局态）；
- 在你“最多两光子、无暗计数、click 吸收光子”的模型下，最多两次 click，绝不会出现 4 次。

### 步骤 A：逐 bin 顺序测量（不要用“独立抽样”）

对每个 bin (n)，对该 bin 的两个输出端口（对应链上的两个 site）做一次联合 POVM 采样：

$$
{K_\mu^{(n)}}*{\mu}
\quad\Rightarrow\quad
\mu_n\sim p*\mu^{(n)}=\langle\Psi|K_\mu^\dagger K_\mu|\Psi\rangle
$$
$$
|\Psi\rangle \leftarrow \frac{K_{\mu_n}^{(n)}|\Psi\rangle}{\sqrt{p_{\mu_n}^{(n)}}}
$$

在代码层面，这个“抽样 + 施加 + 归一化”正是 `apply_two_site_kraus()` 在做的事。

> **关键点：你必须用更新后的 MPS 继续处理下一个 bin。**
> 只要做到这点，“第一次 click 会影响第二次 click 的分布”自动发生。

### 步骤 B：记录 click，但 click 数到 2 就可以停（有条件）

你现在代码里是 `if len(clicks) >= 2: break`。这段逻辑**不是必然错**，但成立有前提：

> 两次 click 对应的 Kraus 必须真的是“吸收型”（把光子映到 vacuum），这样两次 click 后场就是真空，剩余 bins 自动因子化。

如果你将来加入暗计数、非吸收型模型或更复杂探测器效应，这个早停就可能变成真问题（因为你并没有把“后续 bins 都 no-click”也条件化进去）。

**最稳健的做法**：停之前加一个检查，比如计算剩余场的总光子数期望，如果已经 ~0 才停；否则继续把后续 bins 也做 no-click 更新。

### 步骤 C：测完（或早停后确认真空）再取两原子态、trace 光子

最后原子态就是对光子自由度做偏迹：

$$
\rho_{AB}=\mathrm{Tr}*{\text{photons}}\big(|\Psi*{\text{cond}}\rangle\langle\Psi_{\text{cond}}|\big)
$$

你代码里就是 `mps.get_reduced_density([0,1])`，它在数学上是完全正确的约化态。

------

## 6) 顺便把你之前问的 `extract_spin_state()` 定性：它是“后验”还是“先验”？

它本身不是 BSM，只是：

1. 取两原子约化态 $9\times 9$；
2. 抽取 $|0/1\rangle\otimes|0/1\rangle$ 的 $4\times4$ 子块；
3. **再把这个子块按自身迹重新归一化**。

所以它是不是“后验”，取决于你喂进去的全局态是不是已经被探测记录条件化过：

- 你喂的是“测量前的全局态” ⇒ 它只是一个先验诊断（并且第 3 步引入了额外后选：只看原子落在 qubit 子空间的分支）。
- 你喂的是“逐 bin Kraus 探测后、已经条件化到某个 click 记录的态” ⇒ 它才对应“后验的两原子纠缠程度”。

强烈建议你把它改成返回 `(rho_qubit, p_qubit)`：
`p_qubit = Tr(rho_qubit_before_renorm)`，否则你永远不知道自己在第 3 步又后选掉了多少概率。

------

## 7) 关于“bin 越密，同 bin coincidence 概率趋于 0”的疑惑：你直觉对一半

如果你把“成功”硬绑成“必须同一个 bin”，那在 dt→0 的连续极限里确实会趋于 0。
实验上真正做的是：**固定一个物理 coincidence window (W)**，dt 变小只是把 (W) 切成更多 bins，所以总成功率不归零。

------

## 8) 你现在这次“4 click”我建议你用两条最短路径直接定位 bug

不需要我看你本地更多文件，你自己按下面两条打印一下就能立刻知道问题在哪里：

### (1) 确认你到底跑的是哪个探测函数文件

因为你当前日志和你上传的 `detection.py` 明显对不上。
$你本地跑一下 `print(run_two_photon_detection.__file__)` 这种级别就够了。$

### (2) 每次 click 后检查“总光子数是否减少”

`apply_two_site_kraus` 是真的把 Kraus 作用在态上并归一化的。
所以如果你在第 1 次 click 后总光子数没有下降（比如从 ~2 还是 ~2），那就是：

- 你没有在 click 后用更新后的 MPS 继续（等价于“独立抽样”）；或
- 你的 click Kraus 不是吸收型（没有把光子湮灭）；或
- 你把 Kraus 作用在了错误的 site / 错误的子空间（比如 6D/18D 嵌入错了）。

------

### 最后一句话总结

- **指导思路（逐 bin 的 Kraus 条件化/量子跳跃）是对的**：它天然保证不会抽到物理上概率为 0 的“乱点”，也天然保证第一次 click 会改变后续 bins 的条件分布。
- 你现在出现 4 click，说明你实际执行的探测路径没有做到“每次抽样后更新全局态并吸收光子”，或者你根本没跑到你以为的那份 `run_two_photon_detection`。

如果你愿意把你本地那段打印 “Using 6D Kraus operators (36x36) - optimized!” 的探测函数$或对应 `channels.py`$也贴出来，我可以直接在那份代码上指出是哪一行导致“同一光子被多次计数”。但就算不贴，你照着上面第 5 节把探测逻辑改成“逐 bin Kraus + 真实作用回 MPS”，4 click 这种现象会立刻消失。

