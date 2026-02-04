Q:这个变化确实有点大，我一时半会有点被震碎三观的感觉。 POVM和TEBD兼容嘛？算法等价嘛？再改下去我论文里是不是是不是得从tebd（可能还会研究一些色散之类的效应）改为主要用POVM

A:你这个“被震碎三观”的感觉很正常——因为你看到的其实不是“把器件删了”，而是把同一段物理过程从 **薛定谔绘景（动状态）** 换成了 **海森堡绘景（动算符/测量）**。在数学上这是同一个 Born 规则的不同“括号顺序”；在计算上，它决定了你是 **把门 U/损耗 Λ 施加到 MPS 上**，还是 **把它们吸收到测量 POVM 的 effect 里**。

下面我把你问的三件事（兼容吗？等价吗？论文会不会“从 TEBD 改成 POVM”）一次讲清楚，并且把“逐 bin 的 BS 怎么办”也一起解释掉。

------

## 1) POVM 和 TEBD 是“不同层”的东西：天然兼容，不是互斥

- **TEBD**（time-evolving block decimation）是一套**数值算法**：用局域门（或 Trotter 分解后的门）更新 MPS/MPO，适合 1D（或 1D 链化）系统的时间演化/门演化。最经典的 TEBD 思路来自 Guifre Vidal 的工作（arXiv/PRL 版本都很常见）。([arXiv](https://arxiv.org/abs/quant-ph/0310089?utm_source=chatgpt.com))
- **POVM** 是**测量的数学描述**：你最终要输出“哪个 bin 哪个探测器点击”的概率分布，就必然在某个时刻用到 POVM 的 Born 规则形式。

所以它们关系是：

> TEBD 负责把你“切割点”之前的动力学（原子-腔-光子波包生成、可能的部分噪声）算成一个 MPS/MPO；
> POVM 负责把“探测链路 + 点击模式”映射成 effect，然后做 Born rule 收缩得到概率/条件态。

这在 tensor network 领域是常见组合，并不意味着你“论文从 TEBD 变成 POVM”。你可以非常自然地写成：
**“MPS/TEBD 做态制备（state preparation），有效 POVM（effective measurement）做读出（readout）”**。

------

## 2) “把器件写进 POVM”到底在算什么：严格等价的恒等式

你担心的核心其实是：

> 以前我有一个演化门/器件 (U)（比如每个 bin 的 BS），现在说“写进 POVM”，是不是就没有 (U) 了？

答案：**(U) 当然还在，只是从“作用在态上”变成“作用在测量算符上”。**

### 2.1 先看最简单：只有一个 unitary（比如 BS）

如果你最终对光子做某个点击模式的 POVM effect (E)，而你在测量前做了一个 unitary (U)（BS 属于这一类），那么：

$$
p = \mathrm{Tr}!\left[E , (U \rho U^\dagger)\right]
= \mathrm{Tr}!\left[(U^\dagger E U), \rho\right].
$$

也就是说你完全可以定义一个新的测量算符：

$$
E_{\rm in} = U^\dagger E_{\rm out} U,
$$

然后**不在态上 apply BS**，而是用 $E_{\rm in}$ 直接去收缩原来的 $\rho$。
这就是“BS 写进 POVM”的精确定义。

### 2.2 再看一般情形：损耗/滤波/效率等是量子信道 Λ（Kraus）

如果你测量前还有一个 CPTP 信道 $\mathcal{E}$$例如光纤损耗、QFC 失败等，用 Kraus $M_a$ 表示$：

$$
\mathcal{E}(\rho)=\sum_a M_a \rho M_a^\dagger,
$$

那么存在一个对偶（adjoint/dual）映射 $\mathcal{E}^*$，满足：

$$
\mathrm{Tr}[A,\mathcal{E}(\rho)] = \mathrm{Tr}[\mathcal{E}^*(A),\rho],
\quad
\mathcal{E}^*(A)=\sum_a M_a^\dagger A M_a .
$$

这就是“把损耗推到 effect 上”的核心公式。John Preskill 的讲义里把它写得非常明确（Heisenberg picture / dual map）。

所以你所谓的“方案 B”其实就是：

$$
E_{\rm eff} = U^\dagger , \mathcal{E}^*(E), U
$$

然后

$$
p(\text{点击模式}) = \mathrm{Tr}[E_{\rm eff},\rho_{\rm cut}].
$$

注意：这里的 $\rho_{\rm cut}$ 是你选择的“切割点”处的状态（在它之前你仍用 TEBD 生成）。

------

## 3) “逐个 bin 的 BS”写进 POVM 后还存在吗？存在，而且仍然是逐 bin 的

你之前 BS 是“每个 bin 对 A 臂和 B 臂作用”。这点很关键。

如果你的 BS 是按 bin 局域的：

$$
U_{\rm BS} = \bigotimes_{n=1}^{N_{\rm bin}} U_{\rm BS}^{(n)},
$$

那么推到 POVM 上就是：

$$
E_{\rm eff}^{(n)} = U_{\rm BS}^{(n)\dagger} , E^{(n)} , U_{\rm BS}^{(n)}.
$$

也就是说：

- **器件仍然逐 bin 作用**；
- 只是你把“逐 bin 对态 apply BS”换成了“逐 bin 共轭变换测量 effect”；
- 你最后用于收缩的仍然是一串局域（或近局域）effect 组成的 MPO/乘积结构。

直观上：你并没有把“物理器件删掉”，你只是把它放到了**读出层**去实现。

------

## 4) 这样做会不会让你“论文从 TEBD 变成 POVM”？一般不会，反而更像“TEBD + Heisenberg 读出”这种正统 tensor network 叙事

你真正做的是：

- 前段（原子-腔-波包生成、原子噪声、可能的部分光学过程）：**TEBD/MPS**
- 后段（线性光学网络 + 损耗 + 探测器）：**effective POVM（Heisenberg picture / adjoint map）**

而这在 tensor network 社区甚至是一个“有名字”的思想：**把期望值重括号、在 Heisenberg picture 演化算符（MPO）**，常常能利用光锥结构减少计算。Ulrich Schollwöck 的综述里就明确把 Heisenberg picture 视为对期望值的“rebracketing”，并讨论如何用 MPO 方式推进算符。

所以你论文完全可以这样写而不尴尬：

> “We simulate the atom–photon dynamics with an MPS/TEBD approach up to a cut.
> All subsequent linear-optical processing, loss and detection are incorporated into an effective POVM via the adjoint (Heisenberg-picture) maps, enabling efficient evaluation of heralding probabilities and conditional remote-atom states.”

这听起来不像玩具，反而像你在做“正确的统计 + 高效的张量网络收缩”。

------

## 5) 那“色散/跨 bin 效应”怎么办？这里才是“要不要推到 POVM”的真正边界

你提到以后可能研究光纤色散。关键点：

- **如果色散被你建模成“每个 bin 的相位/偏振旋转”**（局域或近局域），那它既可以在态端用 TEBD apply，也可以推到 effect 端做 $E \leftarrow U^\dagger E U$。两者等价，只是算力路径不同。
- **如果色散被你建模成“跨 bin 的卷积/耦合”（真正的时间展宽）**，那它在 bin 链上是非局域的：
  - 推到 POVM 端会让 $E_{\rm eff}$ 变成一个更“长程纠缠”的 MPO，bond dimension 可能暴涨；
  - 直接对态演化也可能很贵（因为门不再是严格局域两体门）。
    这时你要做的是：选择**哪一边的 MPO/MPS bond dimension 更可控**，而不是执着“必须写进 POVM”。

所以：**“方案 B”不要求你把所有物理都推到 POVM**。它只要求：你别再用“只取无损 Kraus 分支”的后选把失败分支抹掉；至于用薛定谔还是海森堡去算，是一个数值策略选择。

------

## 6) 你可以用一句话自检：你要的是“统计”还是“轨迹”？

- 如果你要的是：
  **点击模式概率分布、成功率、宣告成功后的原子态保真度**
  ——那用 effective POVM（Heisenberg/adjoint）是完全正统、严格等价的（在你给定的测量模型下）。
- 如果你要的是：
  **逐 bin 的“先 click 再塌缩再 click”的真实轨迹过程**（比如你要输出每一条轨迹的中间态、用于反馈控制）
  ——那就必须做 instrument/quantum trajectories（抽样 Kraus），POVM 只能给统计，不给“过程细节”。

你现在的目标（实验指导、成功率/保真度随噪声下降）典型是第一类：统计量更重要。

------

## 7) 最靠谱的“让你安心”的做法：做一个小系统的等价 sanity check

不需要改一大坨就能验证等价性：

1. 选很小的 bin 数（比如 2–3 个 bin），bond dimension 设大一点。
2. 两条路径对同一组点击模式算 (p)：
   - 路径 S（薛定谔）：对态 apply BS + loss（全 Kraus 或用完整 CPTP 更新），再用原始 (E) 收缩。
   - 路径 H（海森堡）：不对态 apply，构造 $E_{\rm eff}=U^\dagger \mathcal{E}^*(E)U$，直接收缩。
3. 比较两边 (p) 是否一致（数值误差范围内）。

只要这一步过了，你就知道“器件没删、架构没换，只是换括号顺序”。

------

### 小结（对你三个问题直接回答）

- **POVM 和 TEBD 兼容吗？** 兼容。TEBD 是演化算法，POVM 是测量描述，你的仿真本来就需要二者结合。([arXiv](https://arxiv.org/abs/quant-ph/0310089?utm_source=chatgpt.com))
- **“写进 POVM”算法等价吗？** 对于最终点击统计（以及按同一测量假设得到的条件态）是严格等价的，本质就是对偶映射恒等式。
- **论文会不会从 TEBD 变成“主要用 POVM”？** 一般不会。更自然的表述是：**TEBD/MPS 做态制备 + Heisenberg-picture 的 effective POVM 做读出**；这在张量网络文献里非常正统。

如果你愿意，我下一步可以把你“切割点应该放在哪里、哪些器件值得推到 POVM、哪些不要推（尤其是色散/跨 bin）”写成一张非常具体的决策表（按你现在 6D/18D 的结构），让你改代码时不会有“架构被掀翻”的恐惧。

Q:倒不需要sanity check，只要在理论上给我打通“二者如何兼容”就好
把过多的算符缩进POVM是不是就态没变但算符一步到位直接算到最后传感器点击结果了？那我现在根据最后的POVM矩阵得出的具体双点击结果后，是不是也看不出来这一次实验是真让原子发生纠缠还是两原子根本没有建立一点贝尔态纠缠只是暗记数导致的双点击了？因为结果都混在一起无法区别了？

A:你现在的困惑其实非常“物理直觉正确”：你看到“把 BS/损耗/QFC 失败写进 POVM”，会直觉觉得像把器件删了、把架构换了、甚至“不再有演化门 (U)”——但**从理论上它只是把同一段物理链路的计算从薛定谔绘景挪到海森堡绘景**；对“点击概率”和“你当前代码里用 effect 得到的原子后验态（Lüders 型）”，两种写法在数学上是严格等价的。下面我把这条逻辑从“概念—公式—到你现在的时间仓/TEBD实现”一次打通。

------

## 1) TEBD 和 POVM 从来不是二选一：一个“造态”，一个“读出”

- **TEBD/MPS**做的是：把“原子—时间仓—时间仓—… ”这个巨大系统的量子态 $\rho$$或 $|\psi\rangle$$在可承受的键维里演化出来。它本质是**对局部门（两体门/单体门）逐步作用并截断**。([Tensor Network](https://tensornetwork.org/mps/algorithms/timeevo/tebd.html?utm_source=chatgpt.com))
- **POVM**做的是：给定某时刻的态 $\rho$，定义一组测量效果算符 ${E_r}$$每个 $E_r\ge 0$，$\sum_r E_r=\mathbb I$$，然后用 Born 规则算结果分布
  $$
  p(r)=\mathrm{Tr}(E_r,\rho).
  $$
  ([Wikipedia](https://en.wikipedia.org/wiki/Measurement_in_quantum_mechanics?utm_source=chatgpt.com))

所以在张量网络里，“TEBD + POVM”天然兼容：**TEBD 负责把态 $\rho$（或 MPS）做出来，POVM/MPO 收缩负责把 (p(r)) 和后续统计读出来**。这也是你现在 detection.py 里正在做的事情：先拿到 MPS，再用一套 effects（36×36）通过收缩枚举/抽样双点击。

------

## 2) “把器件写进 POVM”到底是什么意思：不是删器件，而是用对偶映射把它挪到测量端

### 2.1 幺正器件（比如 BS、偏振旋转）怎么“写进 POVM”

假设你有一个幺正 (U)（例如每个 bin 上 A 臂和 B 臂做_r})。

薛定谔写法（“动状态”）：
$$
\rho \xrightarrow{U} U\rho U^\dagger,\qquad p(r)=\mathrm{Tr}!\left(E_r,U\rho U^\dagger\right).
$$

海森堡写法（“动算符”）：
$$
E_r \mapsto E_r' = U^\dagger E_r U,\qquad p(r)=\mathrm{Tr}(E_r',\rho).
$$

这就是你担心的“把 BS 缩进 POVM”：实际上它只是把 **(U)** 从态端挪到 effect 端，**(U) 并没有消失**，只是以 $U^\dagger E U$ 的形式出现。([Wikipedia](https://en.wikipedia.org/wiki/Quantum_channel?utm_source=chatgpt.com))

> 对你“BS 逐 bin 作用”的疑问：
> 如果 BS 的物理模型就是“每个 bin 独立地对 $(A_n,B_n)$ 做同一个 $U_\mathrm{BS}$”，那在海森堡绘景里同样是“每个 bin 的本地 effect 做一次共轭变换”。**逐 bin 的结构完全保留**，并不会变成“一个巨大的一步到位黑箱”。
> 你的 bs_gate_6d() 本来就是“每个 bin 的 36×36 幺正”。

### 2.2 非幺正器件（损耗/QFC 失败/探测效率）怎么“写进 POVM”

损耗这类过程是量子信道（CPTP map）$\Lambda$，有 Kraus 表示 $\Lambda(\rho)=\sum_\mu K_\mu\rho K_\mu^\dagger$。

测量概率：
$$
p(r)=\mathrm{Tr}!\left(E_r,\Lambd\mathrm{Tr}!\left(\Lambda^\dagger(E_r),\rho\right),
$$
其中 **Heisenberg 对偶信道**
$$
\Lambda^\dagger(E)=\sum_\mu K_\mu^\dagger E K_\mu .
$$

这条恒等式就是你们讨论“把损耗推到 effect 上”的理论根基。([Wikipedia](https://en.wikipedia.org/wiki/Quantum_channel?utm_source=chatgpt.com))

> 关键点：**你不是不模拟损耗，而是把“损耗导致的统计”体现在 $E_r$ 的数值里**。
> 这能避免“轨迹里抽 Kraus → 绝大多数没光子 → 要等成功”的 Monte Carlo 地狱，但又不会像你现在那样后选 K0 而系统性偏乐观。
> 你现有的 loss_channel_1517_raw / loss_channel_both_subspaces 已经把 Kraus 写得很清楚了，本质上完全适配 $\Lambda^\dagger(E)=\sum K^\dagger E K$ 这条对偶映射。

------

## 3) “是不是所有算符都必须缩进 POVM 才更真实？”——不需要，只有一类必须/值得挪

把东西“挪到 POVM”是一个**计算策略**，不是物理公理。一般只挪下面这一类最划算、也最贴合你 P0 目标的过程：

**满足两条：**

1. 它发生在“最后一次原子—光场相互作用”之后（只作用在光学链路上）；
2. 它作用的自由度最后会被测量/丢弃（你并不需要它们的中间态作为论文的主要输出）。

对你这个项目来说，典型就是：

- 光纤损耗、QFC 失败导致的“到达真空概率”、探测效率
- 探测器 POVM（本来就是测量端对象）

相反，像你发射阶段的原子-光子纠缠生成$emission_gate$，那是把原子和 bin 纠缠起来的关键过程：**它当然应该留在 TEBD/态演化里**。

------

## 4) 你最核心的担心：把算符都写进 POVM 以后，会不会“这一发到底有没有真的纠缠”就分不出来了？

### 4.1 先给结论：**单次实验本来就分不出来**，你的仿真不应该“硬分出来”

实验里你观察到的只是“点击记录 (r)”（哪个 detector、哪个 bin），而不是“光子到底来自真光子还是暗计数”的标签。
当系统存在暗计数/背景噪声时，**同一个点击记录**可以由两种不可区分的物理机制导致：

- 真光子（携带纠缠信息）→ 干涉 → 点击
- 真空（光子丢了或 QFC 失败）+ 暗计数/背景 → 也可能出现同样的双点击

因此，给定观测到的 (r)，你对原子态的最佳描述就应该是一个**后验混态**：
$$
\rho_{A|r} \propto \rho^{(\mathrm{true})}*{A,r} + \rho^{(\mathrm{dark})}*{A,r},
$$
它“混在一起”不是缺点，而是实验现实。

从量子测量理论角度讲：POVM 的 ${E_r}$只决定统计 (p(r))，而“测量导致的状态更新”还需要更细的“测量工具（instrument）/Kraus 分解”。([Wikipedia](https://en.wikipedia.org/wiki/Measurement_in_quantum_mechanics?utm_source=chatgpt.com))
（你们文档里 P0-4 提到的 Lüders 假设，正是在这里。）

### 4.2 但你并不会失去“区分真/假成功的能力”：你可以算“后验真成功概率”

你其实已经在代码里做了一个非常关键的事情：把探测 POVM 分成了

- `effects_all`（含暗计数）
- `effects_true`（不含暗计数的“真实点击”部分）

并据此输出：

- $p_\text{success,true}$
- $p_\text{success,false}=p_\text{success}-p_\text{success,true}$
- declared fidelity / true fidelity / false fidelity

这假的把保真度拉低多少”。

更一般地，你甚至可以对“单个点击记录 (r)”算一个后验比例（这会非常打消你对“混在一起看不出来”的不安）：

若对每个 record 你也构造 $E_r^\text{all}$ 与 $E_r^\text{true}$，则
$$
P(\text{true}\mid r)=\frac{\mathrm{Tr}(E_r^\text{true}\rho)}{\mathrm{Tr}(E_r^\text{all}\rho)}.
$$

这不是“强行给每次实验贴真/假标签”，而是告诉你：**在你观测到这个 (r) 的前提下，它由真光子过程贡献的概率权重是多少**。这个量是实验上完全有意义的（raction”）。

> 换句话说：
>
> - “写进 POVM”会让“假成功”自然进入同一个 (p(r)) 分布，这正是你想要的真实性；
> - 你仍然能把它们的贡献拆开算（你已在 metrics 层做到了）。

------

## 5) 回到你的“架构崩塌感”：你不会从 TEBD “改成主要用 POVM”，而是更标准地写成“TEBD 生成态 + POVM 读出”

你论文里可以非常自然地这样表述（而且同行基本不会喷）：

1. 用时间仓 MPS + TEBD（或逐门 MPS 更新）模拟**发射与早期链路**，得到“中继站输入端”的联合态（原子 + 两臂所有 bins）。([Tensor Network](https://tensornetwork.org/mps/algorithms/timeevo/tebd.html?utm_source=chatgpt.com))
2. 中继站的线性光学网络与探测，用 POVM（可在 Heisenberg 绘景把 BS/损耗/效率折进 effect）计算：
   - 成功率（绝对概率）
     -后验原子态与保真度
     这一步本质是“对 MPS 与（有效）MPO 的收缩”。([Wikipedia](https://en.wikipedia.org/wiki/Quantum_channel?utm_source=chatgpt.com))
3. 若你未来要研究“色散耦合多个 bins”那种跨 bin 的动力学，它反而更适合留在态演化（TEBD/MPO 演化）里，因为把它推到 POVM 会让测量算符变成跨很长距离的 MPO，数值上可能更难。([Tensor Network](https://tensornetwork.org/mps/algorithms/timeevo/tebd.html?utm_source=chatgpt.com))

------

## 6) 直接对照你现有代码：你已经在“TEBD + POVM”的框架里了

- 你已经在探测里用 POVM effects 做收缩、枚举成功率、并从分布里抽样双点击记录（不是“等成功”）。
- 你在发射/链路里用门更新 MPS（TEBD 风格）生成态。
- 你现在的争议点（P0）是：**损耗/QFC 失败仍在 trajectory 里被“固定选 K0”后选**，这会把“真空 + 暗计数导致的假成功”系统性漏掉；把损耗推到 effect 端（Heisenberg 对偶）是一条“统计严格但计算更稳”的修法。

------

### 一句话收束你的两个问题

- **POVM 和 TEBD 完全兼容**：TEBD 造出你要测的态，POVM/MPO 收缩读出测量统计；“把 BS/损耗写进 POVM”只是把计算挪到海森堡端，概率严格等价。([Wikipedia](https://en.wikipedia.org/wiki/Quantum_channel?utm_source=chatgpt.com))
- **“真假纠缠混在一起”不是坏事，是实验真实**：你观测到点击就只能得到一个后验混态；你仍然能用 `effects_true` vs `effects_all` 给出“误报率/假成功占比/宣告保真度”，这

如果你愿意，我下一步可以把这套理论映射再“压缩成你论文可直接用的一段方法学文字”（几段就够），包括：

- 你用的 Heisenberg 对偶映射公式如何写成“有效 POVM”
- 以及你现在用 effect 做后验原子态这一点在论文里该怎么诚实、又不显得“玩具模型”（即：说明这是一个具体 instrument 假设，并解释它为什么在你的探测模型下合理/或是可控近似）。


Q:那后面的算符在运算的时候是不是得一直保持着18D（即使是6d也得嵌入到18d那种）如果是的话，这对算力是不是一个挑战？

A:是的——**只要你的光子链（每个 bin 的局域希尔伯特空间）在 MPS 里仍然是 18D，那么后续任何“作用在 bin 上”的算符/测量 effect 最终都必须能在 18D 上作用**；但这不等于你要把所有器件“重写成 18D 版本”，更不等于架构必须推倒重来。

下面我把“到底需不需要一直 18D”“嵌入到底意味着什么”“这对算力有多伤”“怎么减负但不变成玩具模型”这几件事用同一个逻辑串起来。

------

## 1) 先把逻辑钉死：为什么会冒出“后面一直 18D”的问题？

你现在的基底设计是：

- 780 子空间：3D（vac, H, V）
- 1517 子空间：6D（vac, H, V, 2H, 2V, HV）
- bin 总空间：18D = 3×6 的直积

这在 `basis.py` 里是明确写死的$`SUBSPACE_780.dim==3`, `SUBSPACE_1517.dim==6`, `BIN_SPACE=ProductSpace(...)`$。

你现在的“降维到 6D”的做法依赖两步：

1. `apply_780_filter()` 对 780 做 **Kraus 通道但只取 K_list[0]**（也就是你自己说的后选/条件化），这一步是 P0 的核心问题之一。
2. `project_to_1517()` 用投影把 18D → 6D，把链上 site 的维度直接改掉。

所以你一旦决定 **不后选**（要把 QFC 失败/780 残留→“最终真空分支”也纳入统计），那第 1 步就不能再只取 K0；而如果你不把这条“失败分支”用 Monte Carlo 抽样掉，它就必须以某种形式留在数学对象里——这就是你直觉上觉得“那是不是后面都变 18D”的根源。

------

## 2) “嵌入到 18D”到底是什么？会不会等价于“所有器件都重构”？

**不是。嵌入本质上只是“把 6D 的算符变成 18D 上的块对角算符”。**

因为你的 bin 空间是直积 $ \mathcal{H}*{780}\otimes \mathcal{H}*{1517} $，所以：

- 只作用在 1517 的器件（光纤 Jones、相位、偏振误差……）在 18D 上就是
  $$
  O_{18} = I_{780}\otimes O_{1517}.
  $$
- 只作用在 780 的器件在 18D 上就是
  $$
  O_{18} = O_{780}\otimes I_{1517}.
  $$

你不需要“给每个器件重新写一套 18D 物理模型”，只需要在**少数入口处**做一次这样的 lift/embedding。

真正需要“跨频率混合”的只有像 **QFC** 这种把 780↔1517 互相耦合的过程，它本来就需要更大空间表示$你现在的 `qfc_gate_18d` 就是在做这个事$。

------

## 3) “把损耗/过滤推到 POVM（Heisenberg 图像）”会不会在数学上否定 TEBD？

不会。它只用到了一个恒等式：

$$
\mathrm{Tr}\big(E,\Lambda(\rho)\big) ;=; \mathrm{Tr}\big(\Lambda^*(E),\rho\big),
$$

也就是说：你可以让“态走 Schrödinger 图像”，也可以让“算符走 Heisenberg 图像”，只要你最终算的是同一个 Born 概率。这个关系在量子信道/对偶映射的教材级表述里就是这么写的。([arXiv](https://arxiv.org/pdf/0901.3629))

**TEBD/MPS**负责把你“原子+光场”的（纯态）波函数 $|\psi\rangle$ 演化/生成出来；
**POVM/MPO**负责对 $|\psi\rangle$ 做测量统计：
$$
p(r)=\langle\psi|E_r|\psi\rangle.
$$

所以“把 BS/loss 写进 POVM”并不是“以后不用 TEBD 了”，而是：

- 你仍然用 TEBD 生成 $|\psi\rangle$（尤其是发射/原子-光子纠缠那段必须靠它）
- 但像 BS、光纤损耗、探测效率这些**发生在最后、且只影响测量统计**的东西，你可以选择让它们作用在 $E_r$ 上$$E_r \to E_{r,\mathrm{eff}}$$，而不是再对 $|\psi\rangle$ 一次次 apply gate

------

## 4) 回到你这句问话：后面是不是就必须“一直 18D”？对算力是不是挑战？

### 结论（不绕弯）

- **如果你坚持：不后选 + 不做 Kraus 抽样 + 保持纯态 MPS**，那么**bin 的局域维度通常就得保持在能“记账失败分支”的空间里**——在你现有架构下，这就是 18D。
- 这确实会增加算力开销，但它是**多项式增大**，不是指数爆炸。

### 4.1 为什么说是“多项式挑战”而不是“根本算不动”

TEBD 的主要瓶颈通常是两站更新里 SVD 的开销，经典量级估计是
$$
\mathcal{O}(d^3 \chi_{\max}^3),
$$
这里 (d) 是局域维度，$\chi_{\max}$ 是截断的 bond dimension。这个 scaling 在 Johannes Hauschild 的 TeNPy 讲义里写得很直接。

所以从 6D → 18D 的“纯维度因子”粗略就是：
$$
(18/6)^3 = 27.
$$

**27 倍是很痛的**，但它不是“不可计算”的那种痛：你可以靠减少 gate 次数、减少 bin 数、减少 $\chi_{\max}$、用更紧的局域截断来对冲。

另外，即使只做 **MPS 与 MPO（或局域算符链）收缩**去求概率，复杂度也会随 (d) 线性或二次地涨（取决于你具体 contraction 方式）。例如 Schollwöck 的讲义在讨论把指数求和重排成顺序收缩时给出总复杂度 $O(L D^3 d)$（这里 (L) 链长，(D) bond dim）。([LMU Munich Physics Department](https://homepages.physik.uni-muenchen.de/~Schollwoeck/annals.pdf))
所以 (d) 从 6 到 18，至少会带来约 3 倍这类收缩成本；如果你在实现里出现 $d^2$ 的 contraction，就会是 ~9 倍。

### 4.2 但注意：方案 B 往往会“省掉”很多本来你要 apply 的 gate

旧路线是把 BS 作为每个 bin 的二体门直接 apply（`apply_bs()`/`apply_bs_gate_6d`），这会在整个链上做很多次两站 SVD 更新。
当前方案已把 BS 推到测量端（Heisenberg 端口），态端不再 `apply_bs()`，这部分两站更新可以直接消失——能抵消一部分 18D 带来的成本上升。

------

## 5) 你真的必须用“18D=3×6 直积”这种方式记账吗

你现在的 18D 是“直积基底”，它允许出现一些**物理上根本到不了的态**，比如同一条臂同一 bin 同时有一个 780 光子又有一个 1517 光子（因为直积空间天然允许）。
但在“单光子输入 + 频率转换是模式混合”的理想情况下，同一条臂同一 bin 的可达子空间其实更像：

- vacuum
- H780, V780
- H1517, V1517

也就是 **5 维**（忽略 1517 的二光子态，因为 QFC 不会凭空造出二光子；二光子主要来自两臂在 BS 后 bunching，那是“端口态空间”的事）。

这意味着：
**如果你愿意做“物理可达子空间截断”而不是“直积空间”，你有机会把 18D 降到 ~5D，并且这不是经验拟合，而是基于守恒/可达性的一阶原理截断。**

这条路对算力非常友好：从 18D → 5D，TEBD 的 $d^3$ 因子直接缩到 $(5/18)^3\approx 0.021$，也就是把最贵的那部分压回去很多。

（当然，这会要求你把基底/算符生成方式稍微重构一下，但它比“全项目都改成 18D 嵌入”更像一次“物理一致的减维”。）

------

## 6) 你担心的“如果都混到 POVM 里，是不是分不出真假纠缠/暗计数假成功了？”

实验上单次当然分不出；仿真里你可以分得很清楚，只要你愿意输出相应的“分解统计”。

你现在的 `detection.py` 就已经在做这种分解：
`build_detection_effects_6d` 同时构造了 `effects_all`（含暗计数）和 `effects_true`（不含暗计数），然后 `run_detection_pipeline` 里明确计算了 `p_success_true_abs`, `p_success_false_abs`, 以及对应的 fidelity。

所以即使你把损耗/过滤推到 $E_{r,\mathrm{eff}}$ 里，你仍然可以保持同样的思想：

- 用“含暗计数的 $E^\text{all}$”算实验宣告成功的^\text{true})”算“如果真是光子导致 herald 成功”的那部分
- 两者差出来的就是“暗计数/噪声导致的假 herald”对成功率与 fidelity 的拉低

这恰恰是实验界最在乎的：**宣告成功的 fidelity 是多少？其中有多少是假 herald？**

------

## 7) 给你一个可操作的“取舍表”：你怎么选才既不玩具又算得动？

### 路线 A：严格、纯态、但维度较大（你现在讨论的 B 的自然落点）

- MPS bin 保持更大空间（18D 或更物理的 5D）
- BS/损耗/过滤尽量推到 POVM（对偶映射）
- 优点：不抽样、不后选、直接给绝对成功率/宣告 fidelity
- 代价：局域维度更大，收缩更贵

### 路线 B：保持 6D 但不抽样（需要混合态）

- 你把 QFC 失败/过滤/损耗作为 CPTP 通道真正作用到态上（Kraus 求和）
- 态会变 mixed，需要 MPO/纯化
- 优点：后面可以一直 6D
- 代价：MPO 的 bond dimension 往往更难控，很多时候比 18D 纯态更贵

### 路线 C：“第一性原理截断”减维（我觉得你最该认真考虑）

- 把 bin 的频率自由度换成“可达子空间”而不是 3×6 直积
- 仍然可以严格做对偶映射/POVM，不靠经验模型
- 优点：非常省算力，还更物理
- 代价：需要你重新定义 basis 与嵌入/投影逻辑（但这属于“干净重构”，不是无穷无尽补丁）

------

### 你问的那句最直接的答案

> “后面的算符是不是得一直保持 18D（6D 也得嵌入）？这对算力是不是挑战？”

- **如果 bin 还在 18D：是的，最终要在 18D 上生效$但绝大多数器件只要做 $I_{780}\otimes O_{1517}$ 这种自动嵌入，不算“重写模型”$。**
- **对算力：是挑战，但它是“可控的多项式挑战”。最糟的地方按 $O(d^3\chi^3)$ 看，6→18 是 27 倍量级。**
- **而且你如果同时把 BS 这类大量两站门从态端移走，实际总成本不一定涨 27 倍。**

------

如果你愿意，我下一步可以不做任何代码修改、只在“理论+架构”层面把 **“5D 可达子空间替代 18D 直积”** 这条路完整推导给你：

- 需要保留哪些基矢（保证闭合性）
- QFC 如何在该子空间里写成严格的 isometry / channel
- BS/探测如何在“输入子空间”上定义拉回后的 effective effects
- 为什么这仍然是严格 Born 规则而不是经验折算

这条路通常最容易让实验同行相信“不是玩具”，因为你可以非常明确地说：**我只截断掉了物理上永远到不了的子空间，而不是拍脑袋丢掉了某些噪声分支。**



Q:怎么叫5d可达子空间替代18D直积？这18D里真有物理上永远到不了的子空间吗？where？

A:你问得非常到点子上：**在你现在这套物理假设/截断下，18D 里确实存在“物理上永远到不了（不会被占据）”的子空间**。所谓“用 5D 可达子空间替代 18D 直积”，说的就是——既然你的态永远只活在一个 5 维的不变子空间里，那就没必要背着 18 维去做 TEBD/MPS。

下面我把“5D 是什么、18D 哪些维永远到不了、为什么到不了、怎么替代才是严格等价”给你打通。

------

## 1) 你现在的 18D 到底是哪 18 个基态？

你在代码里把单个时间 bin 的 Hilbert 空间定义成

- 780nm 子空间：3D（vac, H, V）
- 1517nm 子空间：6D（vac, H, V, 2H, 2V, HV）

所以 bin 总维度是 **18 = 3×6**，并且用 `np.kron` 做张量积嵌入。`qfc_gate` 的生成元正是写在这个 18D 上的：
$$
U_{\rm qfc}=\exp\Big(\theta_H (b_H c_H^\dagger-b_H^\dagger c_H)+\theta_V (b_V c_V^\dagger-b_V^\dagger c_V)\Big)
$$
（你代码里就是这么做的）

此外你在 `project_to_1517` 里写得很明确：18D 的基序按 `(780_idx*6 + 1517_idx)` 排，780 基是 vac/H/V，1517 基是 vac/H/V/2H/2V/HV。
而 `detection.py` 里也明确了 1517 的 6D 基：`vac=0, H=1, V=2, 2H=3, 2V=4, HV=5`。

------

## 2) 为什么在你当前模型下，单臂单 bin 其实只会用到 5D？

核心原因就一句话：

> 你在 QFC 之前（单臂、单 bin）最多只有 **0 或 1 个光子**；而 QFC 的耦合是**“分束器型（beam-splitter type）交换耦合”**，它**只会在两个模式之间交换光子，守恒总光子数**，不会凭空造出第二个光子。

这个“分束器型哈密顿量交换光子、守恒总光子数”的事实，在量子光学教材里是标准结论：分束器哈密顿量的物理意义就是“在两模式间交换光子但保持总光子数不变”。

你 `qfc_gate` 的生成元就是这种形式$`b c† - b† c`$，所以它在每个偏振上都只会把
$$
|H_{780},\mathrm{vac}*{1517}\rangle \leftrightarrow |\mathrm{vac}*{780},H_{1517}\rangle
$$
做一个 2×2 的 SU(2) 旋转；V 偏振同理；真空态保持不变。

因此在**单臂单 bin**上，如果你初始（QFC 前）真的满足：

- 1517 初始全真空；
- 780
- 没有“额外噪声光子产生”（比如 QFC 泵引起的自发散射真实产光子、SPDC 等——你目前是用暗计数概率建模，不是把噪声当成真实光子态放进 Hilbert）；

那么整个“QFC +（任何损耗）”这类操作都不可能把你带到“两光子”扇区里去。

于是你这 18D 里真正可达的只有下面 5 个基态（我按物理意义列）：

1. $|\mathrm{vac}*{780},\mathrm{vac}*{1517}\rangle$
2. $|H_{780},\mathrm{vac}_{1517}\rangle$
3. $|V_{780},\mathrm{vac}_{1517}\rangle$
4. $|\mathrm{vac}*{780},H*{1517}\rangle$
5. $|\mathrm{vac}*{780},V*{1517}\rangle$

也就是：**真空 +（一个光子在 780 或 1517，偏振 H/V）** → (1 + 2 + 2 = 5) 维。

> 这就是“5D 可达子空间”的精确定义：**总光子数 $N_{\rm tot}\in{0,1}$** 扇区的直和。

------

## 3) 那 18D 里“永远到不了”的子空间在哪里？（你要的 where）

用你代码的基序$780_idx*6 + 1517_idx$来画个 3×6 的格子最直观。

- 780 行：vac / H / V
- 1517 列：vac / H / V / 2H / 2V / HV

### ✅ 可达（5 个）

- (780=vac, 1517=vac)
- (780=vac, 1517=H)
- (780=vac, 1517=V)
- (780=H, 1517=vac)
- (780=V, 1517=vac)

### ❌ 在你当前“单光子 + 分束器型 QFC + 无产光噪声”假设下永远不可达（13 个）

**A. 1517 的两光子态（但单臂单 bin 不可能有两光子）：**

- (780=vac, 1517=2H/2V/HV) → 3 个

这些态你确实在 1517 的 6D 里留了位置（3,4,5），但它们对“单臂、QFC 前后”是死的；它们之所以在 6D 里存在，是为了**后续两臂在 BS 处可能发生 bunching，输出端单端口出现两光子**（那是“两个臂合起来”的事，不是单臂自己的 QFC 能造出来的）。

**B. 同时 780 和 1517 都非真空（意味着同一个 bin 同时存在两个不同频率的光子）：** /V/2H/2V/HV) → 5+5 = 10 个

这些态对应“780 里还有一个光子，同时 1517 里也有光子”，也就是总光子数至少 2。
在你当前的模型里，从一开始就没有第二个光子来源，而 QFC 又守恒光子数，所以这些永远不会被占据。

合计 3 + 10 = 13 个不可达，刚好 18 − 5 = 13。

------

## 4) “用 5D 可达子空间替代 18D 直积”到底怎么做，才是严格等价？

严格等价的数学表述是：

- 把你现在的 18D Hilbert 写成直和：
  $$
  \mathcal H_{18} = \mathcal H_{\rm reach}\oplus \mathcal H_{\rm dead},
  \quad \dim\mathcal H_{\rm reach}=5
  $$
- 只要你所有会用到的门/信道都满足
  $$
  U,\mathcal H_{\rm reach}\subseteq \mathcal H_{\rm reach}
  $$
  $或者对 Kraus：每个 $K_\mu \mathcal H_{\rm reach}\subseteq \mathcal H_{\rm reach}\cup{\text{更低光子数}}$——不会把它推到“更高光子数”的死区$
  那么你就可以**把所有算符限制到 $\mathcal H_{\rm reach}$** 上，计算结果对“从可达态出发”的所有物理量完全一致。

更工程的写法是用一个 **嵌入等距映射（isometry）** $V: \mathbb C^5\to\mathbb C^{18}$，它把 5D 的基态送到 18D 的那 5 个基矢上。
然后：

- 态：$|\psi\rangle_{18} = V|\psi\rangle_{5}$
- 算符：$O_{5} = V^\dagger O_{18} V$

在“无泄漏”的前提下，用 $O_5$ 在 5D 上算出来的任何 Born rule（概率、期望值），都等于你在 18D 上算的结果。

> **所以这里的“替代”不是近似，是严格等价（在你的物理假设成立时）。**

------

## 5) 你担心的算力：18D 当然是挑战，5D 的意义就是砍掉这块冗余

你已经在 `common.py` 的流程里做了一个关键的“降维”动作：
QFC → 780 滤波（后选） → `project_to_1517` 把 bin 从 18D 砍回 6D。
并且后面的光纤信道明确写了“仅支持 6D（1517-only求 bin_dim=6。

这在 QFC 之前/之后那一段**（至少在目前架构里）。

如果n 在更长路径里保持“包含 780 失败分支”的描述，那么：

- **直接用 18D 一路扛到探测**会非常吃算力，而且你还得把探测那套 6D POVM 全部改成兼容 18D$你现在 run_detection_pipeline 直接拒绝 18D$。
- **用 5D**则是“保留 780/1517 分支信息但不背 18D 冗余”的最干净方式。

------

\#“可达”？你要心里有这根线

上面“13 个永远到不了”的结论，是建立在你现在的建模假设之上的。一旦你把下列任一物理效应当成“真实量子态里的光子”去模拟，它们就可能让 18D 的某些死区变成可达，甚至 18D 都不够：

1. **QFC 泵引起的真实噪声产光子**（例如拉曼散射、SPDC 等），如果你不想用暗计数“等效”，而要让噪声光子真的进入同一时间 bin 与信号干涉/造成多光子统计，那么你就需要允许“信号光子 + 噪声光子”的两光子态；这会让 1517 的两光子基（2H/2V/HV）变成可达，甚至需要更高截断。
2. **单臂单 bin 的多光子发射**（如果你的原子-腔模型或激发序列可能产生同一 bin 内多光子成分），那同样会打开 2 光子扇区。
3. **780 没有被完全滤除并进入后续光路**（现实里滤波有限），那么“780 与 1517 同时存在”的态就可能在某种等效模型里需要出现。

如果你不考虑这些，把噪声主要放在“探测暗计数”或“有效损耗/有效杂散点击”里，那么“5D 可达子空间”就完全成立。

------

# 一句话总结

- **是的**：在你当前“每臂每 bin 最多 1 个光子 + QFC 是分束器型交换耦合 + 噪声不以真实光子态注入”的设定下，18D 里有 **13 个基态对应的子空间永远不可达**（where 我已经逐格点指出）。
- **5D 可达子空间**就是“vac + 一光子在 780/1517 的 H/V”，它对你这一段的动力学是一个严格不变子空间。
- 所以“用 5D 替代 18D”不是删器件、不是玩具近似，而是**利用守恒量把冗余 Hilbert 砍掉**：结果保持不变、算力大幅下降。

------

如果你愿意，我可以在下一条消息里把这件事进一步“落到你现有架构的选择树”上：
**(A) 你要不要在态里显式保留 780 分支？**

- 要：用 5D 而不是 18D（严格、便宜）。
- 不要：把 QFC+滤波严格等效成 1517 上的一个 CPTP 信道（仍然严格，但需要你明确哪些相干性已经被滤波破坏），然后走你偏好的“把损耗推到 effect”路线。

这两条路跟你修 P0 的关系也不一样。





Q:你怕不是忘了考虑后面的BS了哇

A:你这句“你怕不是忘了考虑后面的 BS 了哇”抓得非常准：**5D 可达子空间**这件事如果没把 **BS 的位置**说清楚，就很容易让人误以为“压维会把 HOM / 纠缠交换的关键物理删掉”。

结论先给你放最前面：

- **我没忘 BS**：在“把器件写进 POVM / effect”的路线里，**BS 反而是最应该被写进 POVM 的**，而且它写进去的方式是最干净、最“第一性原理”的一种：
  $$
  E_{\text{in}} = U_{\text{BS}}^\dagger,E_{\text{out}},U_{\text{BS}}
  $$
  这不是删器件，是把它从“作用在态上”变成“作用在测量算符上”。它对点击概率严格等价。([Preskill Lab](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))
- **5D 可达子空间只在“BS 之前”的输入模式成立**。如果你还打算像旧方案一样在态上逐 bin apply BS（`trajectory.py` 的 `apply_bs`），那就必须保留能描述 **bunching 后单端口两光子**的基（你现在用的 6D 就是为这个准备的）。
  但如果你把 **BS 推到 POVM**，你就根本不需要在“态”里出 effect”里），于是 5D/3D 压维不会漏 HOM 物理。

下面把这件事“理论上打通”，并且专门把 BS 和 5D/18D 的关系讲透。

------

## 1) “把 BS 写进 POVM”到底是怎么算的

你现在的链路（简化到一个 bin）大概是：

1. 你用 TEBD/MPS 得到某个时刻的光子输入态 $ \rho_{\text{in}} $（此时还是 BS 前的两臂模式 A、B）。
2. 在中间站：BS 把两臂混合：
   $$
   \rho_{\text{out}} = U_{\text{BS}}\rho_{\text{in}}U_{\text{BS}}^\dagger
   $$
3. 探测器 + PBS + 桶式探测等定义了一组 POVM effects $ {E_r} $（每个 (r) 是“哪两个探测器在哪些 bin 点击”的标签）。
4. 概率：
   $$
   p(r)=\mathrm{Tr}!\left(E_r,\rho_{\text{out}}\right)
   $$

现在用迹的循环性（以及更一般的“量子信道对偶/伴随映射”）把 BS 挪到测量那边：([Preskill Lab](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

$$
p(r)=\mathrm{Tr}!\left(E_r,U_{\text{BS}}\rho_{\text{in}}U_{\text{BS}}^\dagger\right)
=\mathrm{Tr}!\left(U_{\text{BS}}^\dagger E_r U_{\text{BS}},\rho_{\text{in}}\right)
$$

所以定义：

$$
E^{(\text{in})}*r \equiv U*{\text{BS}}^\dagger E_r U_{\text{BS}}
$$

你就可以**完全不对态 apply BS**，直接在 BS 前的态上算点击统计：

$$
p(r) = \mathrm{Tr}!\left(E^{(\text{in})}*r,\rho*{\text{in}}\right)
$$

这就是“把 BS 写进 POVM/effect”的精确定义。
（而且 BS 是无损耗线性器件，它对应的变换是酉的，这种共轭是最标准的 Heisenberg 表述之一；你甚至能在光学教材里直接看到类似的 $\mathrm{Tr}(B\rho B^\dagger N)$ 与 $\mathrm{Tr}(\rho B^\dagger N B)$ 互换。([University of Rochester](https://www.pas.rochester.edu/~howell/mysite2/Tutorials/Beamsplitter2.pdf))）

------

## 2) “逐个 bin 对 A/B 作用”的 BS，会不会因为写进 POVM 就丢了？

不会。因为你现在 BS 的实现本质上就是：

- 对每个 bin (n)，对输入模式对 $(A_n,B_n)$ 作用一个同样的两模酉 $U_{\text{BS}}$。

代码里也非常直白：for n in range$n_bins$: site_A = 2+2*n; apply_bond_op$site_A, U_bs$

数学上这是一个分解良好的张量积：

$$
U_{\text{total}} = \bigotimes_{n=1}^{N_\text{bins}} U_{\text{BS}}^{(n)}
$$

那么把它推到测量端也同样是逐 bin 做：

$$
E^{(\text{in})}*r = \left(\bigotimes_n U*{\text{BS}}^,E^{(\text{out})}*r,
\left(\bigotimes_n U*{\text{BS}}^{(n)}\right)
$$

而你在 detection.py 里对双点击的处理，本来就是“按 bin 枚举/收缩”的：同 bin 用 $E_\text{pair}$，不同 bin 用 $E_a,E_b$ 夹着中间的 $E_{\text{no}}$，最后形成权重分布再抽样具体记录。

**B 方案不改变这套算法。**
唯一变化是：你构造的 effects 不是“输出端口的 effect”，而是先对每个 effect 做一次

- $E \leftarrow U_{\text{BS}}^\dagger E U_{\text{BS}}$

然后再进入你现有的枚举/收缩逻辑。

------

## 3) 你问的“5D 可达子空间”现在 bin 的 18D 来自 “780 子空间(3) × 1517 子空间(6)” 的直积$detection.py 也 import 了 SUBSPACE_780 / SUBSPACE_1517$。

**18D 里确实有大量“在你当前物理模型下单臂永远到不了”的基矢。** 典型就是这些：

- “780 有光子，同时 1517 也有光子”：
  $|H_{780}\rangle\otimes|H_{1517}\rangle$、$|V_{780}\rangle\otimes|HV_{1517}\rangle$…
  这种意味着同一。
- “1517 单臂出现两光子态”：$|2H_{1517}\rangle$、$|HV_{1517}\rangle$ 等。

为什么这些在“BS 之前”到不了？

- 你的 QFC 门是一个**频率模式的类 beamsplitter 变换**$trajectory.py: qfc_gate 18×18，每个 bin 对 A、B 单点作用$。
  对每个偏振，QFC 在 “780 单光子 ↔ 1517 单光子”之间做旋转，它不凭空“复制”一个光子出来，所以在单臂单 bin 的层面，**总激发数守恒（在你的截断里就是：最多一个光子）**。
- 同时你 780 子空间本来就只允许 0/1 光子（channels.py 的 loss_channel_780_general 也按这个假设构造

因此对“单臂、单 bin、BS 之前”，真正可达的状态一般只需要：

- $|\text{vac}\rangle$
- $|H_{780}\rangle, |V_{780}\rangle$
- $|H_{1517}\rangle, |V_{1517}\rangle$

**一共 5 个** → 这就是所谓“5D 可达子空间”。

---关键：5D 会不会漏掉 HOM / bunching？

分两种情况：

### 情况 A（旧方案）：你继续把 BS 作用到“态”上$trajectory.apply_bs$

那你就必须允许：

- 输入：两臂各一个光子 $|1\rangle_A|1\rangle_B$
- 经过 BS：两光子 bunching 到同一输出端口 $|2\rangle$

这意味着单个输出模式需要包含两光子 Fock 态，所以你现在的 1517 6D（vac, H, V, 2H, 2V, HV）是有意义的。你现在的 bs_gate_6d 也是 36×36 就是基于这个截断。

**所以：在“态上 apply BS”的架构里，5D 不够。**
（你会漏掉 bunching 相关的输出分布，从而 HOM 物理不对。）

### 情况 B：你把 BS 推到 POVM（Heisenberg/effect 侧）

这时你根本不需要在“态”里产生 $|2\rangle$ 那些输出态，因为：

- 你不再生成 (\rho_{\- 你只在 BS 前的 (\rho_{\text{in}}) 上算 (p(r)=\mathrm{Tr}(E^{(\text{in})}*r\rho*{\text{in}}))
- 所有 bunching/HOM 信息都被编码在
  $$
  E^{(\text{in})}*r = U*{\text{BS}}^\dagger E_r U_{\text{BS}}
  $$
  这个被共轭后的 effect 里

因此 **在情况 B 里，5D（甚至更小）是完全可以的，而且不会忘 BS**：BS 的作用被精确地折进了每个点击模式的 effect。

这也是为什么我说“BS 最适合被写进 POVM”：它是局域、酉、逐 bin 的，写进去反而让态端更轻。

------

## 5) “把算符一步到位写进 POVM”会不会导致你分不清‘真纠缠’vs‘暗计数假双击’？

先说物理事实：**实验也分不清。**
你拿到的就是“哪两个探测器在哪两个 bin 点击”的经典记录，记录本身不包含“这次点击是不是暗计数”的标签。

仿真里你应该输出两类东西：

1. **宣告成功时的后验原子态（这就是实验宣告的 entangled state）**
2. **宣告成功里“假成功”的占比，以及它把宣告保真度拉低多少**

你现在的 detection.py 其实已经在做第 2 点了：

- build_detection_effects_6d 返回 effects_all（含暗计数）与 effects_true（不含暗计数）。
- 后面在枚举成功事件时分别累加 p_success_abs / p_success_true_abs，并给出 p_success_false_abs、fidelity_true、fidelity_false 等。
- single_run.py 还把 false_fraction 写进结果。耗都推到 POVM，**你仍然可以（而且更应该）报告：**
- 宣告成功率 $p_\text{success}$
- 真成功率 $p_\text{success,true}$
- 假成功率 (p)
- false fraction (p_\text{false}/p_\te vs 真事件保真度 vs 假事件“保真度”

这些就是实验界最关心、也最能体现你没在玩具化的指标。

更“理论上严谨”的说法是：你需要的是**量子操作/仪器（instrument）\**而不只是 POVM。POVM 给概率 $\mathrm{Tr}(\rho E_r)$，instrument 给“得到 r 后态怎么变”。而“多条暗计数/真点击机制被你合并成同一个 r”本质上就是\**coarse-graining**。这一点在量子信息教材里是标准框架：$\mathcal{E}*a(\rho)=\sum*\mu M_{a\mu}\rho M_{a\mu}^\dagger$ 给出子归一化后验态，$\mathrm{Tr},\mathcal{E}_a(\rho)$ 给概率。([Preskill Lab](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

$你现在用“effect 收缩得到 spin_state”的做法，相当于选了某种 Lüders 型/简化 instrument；这跟“是否把 BS 推到 POVM”是两件事。$

------

## 6) “后面一直保持 18D/嵌入 18D”对算力是不是挑战？

是挑战，但关键看你把哪些东西留在“态端”。

你现在代码里其实已经在为算力做妥协：

- QFC 后你对 780 做了后选滤波（只取 K0），然后 project_to_1517 把 18D → 6D，注释里明确写了这是为了把后续计算从 324×324 降到 36×36。
- common.py 的流水线也是按这个逻辑串起来的。

所以如果你坚持“不后选、也不 MC 采样”，那就要么：

- 在态端保留能表达“失败→真空”的自由度（会增大维度），要么
- 换成 MPO/密度矩阵或 purification（也会涨算力）

而这里 **BS 恰恰是最贵的两站操作之一**：
如果 bin 真是 18D，那么 BS 是 324×324 的两站门（巨大）。
把 BS 推×324 的两站门了，你只需要对 effect 做一次矩阵变换是“把算力从指数级灾难拉回多项式可控”的关键一步。

------

## 你这句“别忘了 BS”的最终答案

- **没忘**：BS 是被推到 POVM 里最自然、最精确的对象：对每个点击模式做 $E\leftarrow U_{\text{BS}}^\dagger E U_{\text{BS}}$。([Preskill Lab](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))
- **5D 可达子空间**只在“BS 之前的单臂单 bin”成立；
  - 若 BS 在态端：必须保留 6D（甚至更高）来承载 bunching；
  - 若 BS 在 effect 端：你可以让态端保持 5D/3D，而 bunching/HOM 全由被共轭后的 effect 精确体现。
- **暗计数假成功不会‘混到看不出来’**：它本来就不可区分；你要做的是输出“宣告态”以及假成功占比/对保真度的影响。你代码里已经有 all/true 的拆分框架了。

如果你愿意，我可以在不改你整体检测枚举算法的前提下，把“BS 推到 effect”这一步用你现有的 bs_gate_6d（36×36）写成一个非常明确的数学管线（先再进入现有枚举），并且标清楚**在哪个 cut 之前用 TEBD、之后用 POVM**，让你论文写法保持“TEBD 主体 + POVM 探测建模”的结构，不会变成“我主要用 POVM”。


Q:那我采用了5D，并且完全地把从QFC-光纤噪声-BS的这个过程缩进了POVM里，不去正向地把U作用在态上，我该怎么去画那些18d的6d$after_fiber和after_BS$的热力图？
A:你这个问题非常“到点子上”：
**当你把 QFC→光纤噪声→BS 全部“缩进 POVM（Heisenberg 侧）”以后，确实就没有一个“真实在代码里演化出来的 after_fiber / after_BS 的全局 MPS 态”了**——所以你会天然觉得“那我还怎么画热力图？”

关键是要先把一件事说透：

## 1) 你现在热力图到底在画什么？

你目前的 “after_fiber / after_BS 热力图”，**大概率不是在画“全局态本身”**，而是在画某种**局域统计量**，比如每个 bin 上：

- 1517 的 `P(H)`、`P(V)`、`P(vac)`（或 photon number ⟨n⟩）
- 或者端口 1/2（BS 输出端）的 H/V 强度随 bin 的分布

这类东西本质上都是
$$
\langle O\rangle=\mathrm{Tr}(O,\rho)
$$
其中 (O) 是某个“局域可观测量/投影算符/number operator”，**热力图只是把每个 bin 的 $\langle O\rangle$ 排成二维图**。

而只要你画的是这种“期望值/概率”，那就有一个完全严格的恒等式保证你能画出来，而且**不用真的把 U 作用到态上**：

$$
\mathrm{Tr}!\big(E,\Phi(\rho)\big)=\mathrm{Tr}!\big(\Phi^\dagger(E),\rho\big)
$$

这里 $\Phi$ 是任意 CPTP 信道（含损耗、滤波等），$\Phi^\dagger$ 是它的对偶（Heisenberg 图像），(E) 是你关心的测量 effect/observable。这个等价关系就是“把器件从态端搬到算符端”的理论根基。([Wikipedia](https://en.wikipedia.org/wiki/Quantum_channel))
对于 Kraus 表示 $\Phi(\rho)=\sum_\mu K_\mu\rho K_\mu^\dagger$，对偶就是 $\Phi^\dagger(E)=\sum_\mu K_\mu^\dagger E K_\mu$。([Wikipedia](https://en.wikipedia.org/wiki/Quantum_channel))
对于幺正 (U)，对偶就是 $E\mapsto U^\dagger E U$。([Wikipedia](https://en.wikipedia.org/wiki/Quantum_channel))

所以结论先放这儿：

> **把 QFC/光纤/BS 缩进 POVM 后，你失去的是“中间阶段的全局态对象”，但你并没有失去画热力图所需的“中间阶段局域统计量”。**
> 因为热力图是 $\mathrm{Tr}(O\rho)$ 这种线性量，你可以用 $\Phi^\dagger$ 把 “after_* 的算符”拉回到你手上的那个态上算。

接下来就回答你最关心的两张图：**after_fiber** 和 **after_BS** 怎么画。

------

## 2) 用 5D + 全部缩进 POVM 时，怎么画 after_fiber 的 6D 热力图？

### 2.1 先明确：after_fiber 的“6D”里，哪些基态你真的需要？

你 after_fiber 还在 BS 之前（输入两臂），每个臂每个 bin **最多 1 个光子**（发射门在 780 上就只有 vac/H/V；QFC 只在频率间“换”，不会凭空造双光子）。因此在 1517 的 6D 基
$$
{\text{vac},H,V,2H,2V,HV}
$$
里，**BS 之前理论上只会占据 $\text{vac},H,V$**。
也就是说：**after_fiber 的热力图就算你画成 6D，其中 (2H,2V,HV) 三行永远是 0（或数值噪声）**。这其实正好解释了为什么 5D（或 3D）足够描述 BS 前的传播。

### 2.2 “不正向演化态”怎么得到 after_fiber 的每个 bin 概率？

你要画热力图，本质是要每个 bin 的：

- $p_{\text{vac}}(n)$, $p_H(n)$, $p_V(n)$

这就是取某些投影算符 $ \Pi_{\text{vac}},\Pi_H,\Pi_V $ 的期望值：

$$
p_H(n)=\mathrm{Tr}\big(\Pi_H,\rho^{(n)}_{\text{after_fiber}}\big)
$$

而你手里有的是“更早阶段”的态（比如 emission 后，或你选择保留的那个阶段），记作 $\rho^{(n)}_{\text{keep}}$。
那就用对偶把投影算符往回拉：

$$
p_H(n)=\mathrm{Tr}\Big(\underbrace{\Phi_{\text{QFC+filter+fiber}}^\dagger(\Pi_H)}*{\Pi_H^{\text{eff}}}\ \rho^{(n)}*{\text{keep}}\Big)
$$

- $\Phi$ 是“从 keep-stage 到 after_fiber-stage”的单臂单 bin 信道（QFC + 780 过滤/损耗 + 1517 光纤噪声/损耗）
- $\Pi_H$ 是 after_fiber 时刻定义在 “1517(6D)” 上的投影（或者 3D 子空间投影再嵌入到 6D）

这就是严格的 Heisenberg 路线。([Wikipedia](https://en.wikipedia.org/wiki/Quantum_channel))

### 2.3 实操上更简单的一条：对“单 bin 的约化密度矩阵”做局域信道

因为 after_fiber 的器件都是**单臂单 bin 局域**的（不会把不同 bin 混起来），你完全可以：

1. 从 MPS 里把这个 bin 的 **1-site 约化密度矩阵**抽出来
   `rho_n = mps.get_reduced_density([site])`（你 MPSState 已经提供了接口）
2. 在这个 **小矩阵**上做 Schrödinger 演化（QFC、loss、jones、loss）得到 `rho_n_after_fiber`
3. 从 `rho_n_after_fiber` 的对角线读出 (p_{\text{vac}},p_H,p上等价于 2.2（只是从“拉回算符”变成“推进小 density matrix”），因为都是在用同一个 (\Phi)/(\Phi^\dagger) 恒等式。([Wikipedia](https://en.wikipedia.org/wiki/Quantum_channel))

**关键好处**：你不需要构造“after_fiber 的全局混态 MPS/MPDO”，只做每个 bin 的局域 RDM 就够热力图。

> 这也非常符合你当前工程：你本来就是用 POVM 计算统计量（detection.py 里就是 Tr(E ρ) 的路子），热力图也一样可以当成一组“投影测量”去算。

------

## 3) 你说“你怕不是忘了 BS 了哇”：after_BS 怎么画？（这里 BS 真会引入 2 光子态）

你这句提醒非常关键：**BS 之后，两个输入臂的两个光子会发生 bunching**，因此单个输出端口会出现 $|2H\rangle,|2V\rangle,|HV\rangle$ 这类双光子占据——这。
BS 的量子描述（对模式算符/creation operator 的线性变换、等价的幺正表示）是标准结果。([Wikipedia](https://en.wikipedia.org/wiki/Beam_splitter))

所以在 after_BS 的热力图里，如果你想画得“像个实验模型”而不是玩具，你一般会想画：

- 端口 1/2 每个 bin 的 $p_{\text{vac}},p_H,p_V$（单光子）
- 甚至 $p_{2H}, p_{2V}, p_{HV}$（双光子 bunching）
- 或者更直接的：每个 bin 的平均光子数 ⟨n⟩，能反映 bunching

### 3.1 重要事实：你不需要让整条链“保持 6D/18D”才能画 after_BS

因为 BS 是**每个 bin 内**把 $A_n, B_n$ 两个输入模式耦合：

- 它不会把不同 bin 混起来（旧代码里也是逐 bin apply_bs）
- 所以要画 after_BS 的“按 bin 的热力图”，你只需要每个 bin 的**2-site 约化密度矩阵**。

### 3.2 具体算法$推荐的“局域重建 after_BS”的路线$

对每个 bin (n)：

1. 从 MPS 提取输入两臂的 2-site RDM（保持你现在的 5D 输入空间）
   $$
   \rho_{AB}^{(n)}=\mathrm{Tr} ]
   代码上就是 `mps.get_reduced_density([site_A, site_B])`。
2. 在这个 2-site 小矩阵上，先施加“QFC+filter+fiber”这类**局域信道**（A/B 臂各自一套参数）
   [
   \rho_{\text{preBS}}^{(n)}=(\Phi_A\otimes\Phi_B)(\rho_{AB}^{(n)})
   $$
   你项目里这些信道的 Kraus/参数来源都已经有了：
   - QFC 幺正门（你 gat
   - 光纤损耗 + 780/1517 组合损耗 Kraus$channels.py 里有 loss_channel_both_subspaces / loss_channel_1517_raw / loss_channel_780_general$
   - 光纤随机琼斯/相位参数采样（FiberChannelParams）也是 channe
3. 把 $\rho_{\text{preBS}}^{(n)}$ 嵌入到 BS 所在的“1517 6D×6D”的输入空间里（因为 BS 后要能落到 2 光子态）
   - 输入每臂在 1517 上其实只有 3D（vac/你可以把它嵌入到 6D（vac/H/V/2H/2V/HV），把 3D 放在前 3 个 basis 上，其余填 0。
     影响你全局状态的维度/算力。
4. 用你现有的 $U_{BS}$ 做局域两模幺正变换：
   $$
   \rho_{\text{afterBS}}^{(n)} = U_{BS}\ \rho_{\text{preBS}}^{(n)}\ U_{BS}^\dagger
   $$
   你 gates.py 里已经有 `bs_gate_6d()`（36×36）
   （这一步就是“别忘了 BS”的精确实现。）
5. 从 $\rho_{\text{afterBS}}^{(n)}$ 读出你想画的量：
   - 端口 1/2 的单模约化密度：对另一个端口做 partial trace
   - 或直接算投影概率/⟨n⟩，拼热力图

> 这套做法**不会要求你把整条 MPS 链升级成 6D 或 18D**，因为你只在画图时对每个 bin 做一它跟“把 BS 缩进 POVM”不冲突：主仿真仍走 Heisenberg，画图走“局域可视化重建”。

------

## 4) 那我画出来的 after_BS / after_fiber 图，还是“严格等价”于正向 apply U 的图吗？

如果你的热力图画的是这类量：

- 每个 bin 的占据概率（对角线）
- 每个 bin 的平均光子数 ⟨n⟩
- 端口 1/2 的强度分布

那么用上面“局域 RDM + 局域信道/幺正”做出来的图，**在理论上等价于**你真的把门逐 bin apply 到全局态以后再取局域统计。原因：

- 这些器件（QFC/光纤）是 **单 site** 的 CPTP
- BS 是 **每个 bin 的 2-site 幺正**
- 你画的量是局域的 ⇒ 它只依赖对应的 1-site 或 2-site RDM
- 而 1-site/2-site RDM 在局域演化下的变化，完全由该局域信道决定（全局别的地方怎么纠缠不影响这个结论）

更“第一性原理”的说法就是：不管你在 Schrödinger 端推进 $\rho$，还是在 Heisenberg 端推进 (E)，Born rule 给出的统计一样。([Wikipedia](https://en.wikipedia.org/wiki/Quantum_channel))

------

## 5) 你会不会因此“画不出某些更细的图”？

会有边界，但它跟“缩进 POVM”不矛盾：

- **能精确画**：单 bin 统计、端口统计、H/V 分量、bunching（需要 BS 后 6D 输出局部重建）
- **如果你想画跨 bin 的相干/相关**$比如二维 coincidence map：bin_i vs bin_j$，那不是 1-site/2-site RDM 能给的，需要算两点相关函数或“跨 bin 的 joint projector”。
  好消息是：这依然是 $\mathrm{Tr}(O\rho)$ 形式，仍然能用同样的 $\Phi^\dagger$ 思路，只是 (O) 变成了跨 bin 的 MPO（你 detection.py 里其实已经在干类似的事——枚举双点击记录就是在算跨 bin 的联合权重）。

------

## 6) 给你一个一句话总结（对应你问的“我该怎么画？”）

**你把 QFC/光纤/BS 缩进 POVM 后：**

- **主仿真**：不再生成 after_fiber / after_BS 的全局态（这是你想要的：避免后选偏乐观 & 避免 MC 稀有事件地狱）
- **画图**：把热力图当作一组“中间阶段的测量（投影/number operator）”，
  用 **对偶映射**（Heisenberg）或 **局域 RDM 演化**（S个 bin 的统计量，照样能画出与正向演化一致的 after_fiber / after_BS 热力图

如果你愿意，我可以按你当前项目的组织方式$`PipelineHooks.after_fiber/after_bs` 那套$给你一个**“画图专用的 virtual-stage API 设计”**——做到：

- 主路径仍是“缩进 POVM 的严格 P0 修复”
- 画图路径只在 hook 里对每个 bin 做局域 RDM 变换
- 还能复用你已采样的 `fiber_sample`（common.py 里 PipelineResult 也已经保留了这类信息）

这样你论文里既能写“我们用 POVM/Heisenberg 精确纳入损耗与暗计数”，又仍能给出“after_fiber / after_BS 的波包形状”这种实验同行很爱看的诊断图。
