Q:假如真如你说的那样把 BS 推进 POVM（Heisenberg 端），就可以让态端保持5D（vac H780 V780 H1517 V1517 ），那我把至少要6D（vac H V 2H 2V HV）才能描述完全的BS推进POVM里以后，这个门到底是多少维的？门怎么乘？
A:你这个疑惑非常“对症”：**输入态用 5D（每臂每个 bin 只有 0/1 个光子）**，但 **BS 的物理会把“两臂各 1 个光子”的振幅变成“同一输出端口 2 光子”的振幅**（HOM bunching），所以你直觉上会觉得“没有 2 光子态就没法表示 BS”。
关键是：**把 BS 推到 POVM（Heisenberg 端）并不是“不要 2 光子态”，而是“不要把 2 光子态放进 \*态的表示\* 里”**——它会被编码进 **测量算符（effect）** 里。

下面把“维度到底是多少、怎么乘”用最干净的账本打通。

------

## 1）先分清：你说的 6D，是“输出端口的截断福克空间”，不是“输入态必须的局域维度”

你代码里定义的 1517 单端口 6D 基是：

- `vac, H, V, 2H, 2V, HV`

这 **6D 是为了在一个端口里允许最多 2 个光子**（同一端口发生 bunching 时必须用到）。
因此 **两端口（BS 的两输出口）联合空间是 6×6=36 维**，你也确实把 BS 写成了 `36×36`：`bs_gate_6d()` 。

但注意：这并不意味着**输入态**也必须是 6D。
输入态在 BS 之前，如果物理上保证“每臂每个 bin 最多 0/1 个光子”（这是你想用 5D 的前提），那么**输入每端口其实只需要 3D：`vac, H, V`**（只需要单光子子空间）。

------

## 2）为什么“输入每端口 3D/5D”仍然能正确包含 HOM（bunching）？

因为 HOM 的本质是：**BS 在算符层面是线性变换**，它把输入的产生算符变成输出产生算符的线性组合。
例如（反射率 η 的 BS），在产生算符上有类似变换 ([arXiv](https://arxiv.org/pdf/1711.00080))：

- $ \hat a^\dagger \mapsto \sqrt{1-\eta},\hat a^\dagger + \sqrt{\eta},\hat b^\dagger $
- $ \hat b^\dagger \mapsto \sqrt{\eta},\hat a^\dagger - \sqrt{1-\eta},\hat b^\dagger $

当 η=1/2（50/50）时，两光子输入 $\hat a^\dagger \hat b^\dagger|0\rangle$ 的输出会出现 **$\hat a^{\dagger 2}$** 或 **$\hat b^{\dagger 2}$** 的项（也就是“同一端口 2 光子”），从而产生 HOM 的“无符合”现象 ([arXiv](https://arxiv.org/pdf/1711.00080))。

所以：

- **输入态**可以只包含 “每端口 0/1 光子”的 3D（或你说的含 780/1517 的 5D）；
- **但 BS 作用后的输出态**确实会占用 “每端口最多 2 光子”的 6D；
- 若你不在态端显式生成输出态，而是把 BS 推到测量端，那么 **输出端口的 2 光子信息会进入测量 effect**。

------

## 3）把 BS 推到 POVM 后：门到底是多少维？

你需要分两层回答：

### A. “物理上/数学上 BS 的幺正 U 是多少维？”

在你目前的探测建模（允许同一端口最多 2 光子）下：

- 单端口：6D$`vac,H,V,2H,2V,HV`$
- 两端口：36D
- 所以 **$U_{\rm BS}$。

这点不会因为你把它推到 POVM 就改变：**U 还是那个 U**，做共轭。

### B. “真正进入 MPS 收缩的算符(E_eff)是多少维？”

如果你输入端每个端口只允许 0/1 光子$也就是只用 3D：`vac,H,V`$，那输入联合空间是 3×3=9 维。
把 BS 推到测量端后，你最后会得到一个 **作用在输入 9D 子空间上的有效 POVM**：

$$
E_{\text{in}} ;=; P_{\text{in}}; U_{\rm BS}^\dagger; E_{\text{out}}; U_{\rm BS}; P_{\text{in}}
$$

- $E_{\text{out}}$：你原来在“BS 后的输出端口”定义的探测 effect（36×36）。你现在就是这样造的：`build_detection_effects_6d()` 返回每个点击模式的 `36×36` effect 。
- $U_{\rm BS}$：36×36（见上）。$P_{\text{in}}$：把 36D 空间投影到“每端是 `diag(1,1,1,0,0,0)`，两端口取张量积。

投影后，**你真正用于“态端收缩”的 E_in 可以压缩成 9×9**（因为投影会把其余维度砍掉）。

> 这就是你要的答案：
>
> - **BS 的 U 仍然是 36×36**（因为它必须知道 2 光子 bunching）
> - **但你在 MPS/TEBD 里不再承受 6D/36D 的态演化**，而是用一个 **压缩后的 9×9（或进一步扩展到 25×25）effect** 去算概率。

------

## 4）那你坚持“态端 5D：vac, H780, V780, H1517, V1517”时怎么办？

你 5D 的本质是：**“单光子在两个频段之间占据的可达子空间”**。
这时每臂每 bin 是 5D，两臂联合就是 25D。

关键点：**BS 和电信探测器只作用在 1517 的模式上**。
所以你可以这样做（概念上最干净）：

1. 先在“1517 的输入 3D×3D=9D 子空间”上得到 $E_{\text{in}}^{(1517)}$（上节的 9×9）。
2. 然后把它“嵌入/扩展”为作用在 25D（5×5）输入空间的 effect：
   - 对于包含 780 光子的基矢，它们在“到达 1517 探测器”这条链路上等价于**没有 1517 光子**（只剩暗计数可导致点击）。
   - 这一步本质上是一个**信道的对偶映射**：把“780 被滤掉/不被探测”的过程作为 channel Λ（把 780 态映射到真空态），然后做
     $$
     E_{5D} = \Lambda^\dagger(E_{\text{in}}^{(1517)})
     $$
   - 对偶映射在 Kraus 表示下就是
     $$
     \Lambda^\dagger(E)=\sum_i K_i^\dagger E K_i
     $$
     这是标准恒等式 ([Preskill Lab](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))。

这意味着：

- **你不需要把 BS “改写成 25×25 的门去乘态”**；
- 你只需要得到一个 **25×25 的有效 effect**，然后照 Born rule 去收缩 MPS。

------

## 5）“门怎么乘？”——在 Heisenberg 端就是两种基本操作：共轭 + Kraus 对偶

你把一串器件（QFC、滤波/损耗、光纤、BS、探测）推到 POVM，本质是在做：

- 幺正器件 (U)：
  $$
  E \leftarrow U^\dagger E U
  $$
- 噪声/损耗信道$Kraus ${K_i}$$：
  $$
  E \leftarrow \sum_i K_i^\dagger E K_i
  $$
  ([Preskill Lab](https://www.preskill.caltech.edu/ph219/chap3_15.pdf))

而且顺序是**反过来**的：
如果正向是 $ \rho \to \Lambda_{\rm fiber} \to U_{\rm BS} \to \text{measure }E_{\rm out}$，
那反向就是
$$
E_{\rm pre} = \Lambda_{\rm fiber}^\dagger!\left(U_{\rm BS}^\dagger E_{\rm out} U_{\rm BS}\right)
$$

你现在的探测 effect（含效率和暗计数）其实就是用 Kraus 构造出来的：`E = K†K` 并对同 outcome 求和 ；这和文献里对 on/off 探测器的 POVM 写法一致（效率 η、暗计数 ν 的系数形式）([arXiv](https://arxiv.org/pdf/1206.0824))。

------

## 6）把话说死：你担心的“维度不够导致 BS 无法表示”到底错在哪？

错在把“**输出态需要 2 光子子空间**”误认为“**输入态也必须包含 2 光子子空间** 端真的要把 BS 乘到态上，那当然必须在态端允许 2 光子（所以用 6D）。

- 但你现在讨论的是把 BS 推到 POVM：态端不产生输出态，**测量算符替你把 bunching 的贡献算进去**。
  这时态端保持 3D/5D 完全没问题——你只是把复杂度挪到“预计算 E_eff”的那一步，而不是每次 TEBD 演化都承受 6D。

------

## 7）顺手对照你现有代码：这不是“换架构”，而是“把同一条链路从态端挪到 effect 端”

你目前工程里：

- 6D basis：`SUBSPACE_1517 = ('vac','H','V','2H','2V','HV')`
- 36×36 BS：`bs_gate_6d()`
- 36×36 探测 effects：`build_detection_effects_6d()`
- 探测流水线目前强制 bin_dim==6（意味着你现在确实是“态端到 BS 后再测”这条路径）到 POVM 后，概念上就是：

1. 仍然用 `buil:contentReference[oaicite:22]{index=22}d` 得到 $E_{\rm out}$（36×36）。
   到 $U_{\rm BS}$（36×36）。
2. 做\dagger E U)。
3. 再投影/压缩到输入子空间（9×9 或扩展到 25×25）。
4. 之后 MPS 侧**不再 apply_bs**，而是直接用新的 与抽样。

所以：**U 没有消失**；你只是把它从“TEBD 演化门”变成“POVM 的前置变“9D 投影矩阵 $P_{\text{in}}$ 怎么在你这套基序里写出来（哪几个 index 保留、哪几个置零）”用你现有 `SUBSPACE_1517` 的 index 直接列出来——那样你在实现上会更踏实。

Q:这逻辑根本没有闭环欸？请输出完整的算式，怎么把所有的门都收缩进POVM，这里的维度包括衔接矩阵的维度都是几乘几的。原先色散什么的造就的MPS维度，我觉得把算符塞到POVM里，总不可能消失了不成。这些信息现在在哪里？
请重新看咱的py程序，列出所有所有要收进POVM的门，以及它们的维度(标在矩阵下表里)，并且给出这些维度的算符怎么乘进POVM里最终得出来了能够正确作用于MPS态的维度（检查严格得像一个编译器一样，算式列出来但算符在数学上维数与维数对不上都算逻辑有问题/没打通）

A:你这条要求我完全理解：你不想再听“海森堡绘景等价所以可以”的口头结论，而是要**把整个链路写成一串严格可检查的算式**，并且把每一步的矩阵维度写死，像编译器一样逐项过维数检查——哪里对不上就说明逻辑没闭环。

下面我按你**现有代码**来做（只用你项目里真实存在的空间/门/信道/POVM），并且给出“把哪些门收进 POVM”时每一步的维度变换。
$我会用“每个时间 bin 的两臂 A_n、B_n 形成的一对模式”作为基本单元；多 bin 的情况就是把这些单元按 bin 张量/按你 detection.py 的枚举方式拼起来。$

------

## 0) 先把你项目里“物理空间维度”钉死

这些不是我猜的，是你代码里明确定义的：

- 原子：4 维（|0>,|1>,|e>,|u>）
- 780 子空间：3 维（vac,H,V）
- 1517 子空间：6 维（vac,H,V,2H,2V,HV）
- “bin 格点”（780×1517）：18 维 = 3×6

你 MPS 的链布局（发射结束后）是：
`atomA(4), atomB(4), A1(bin), B1(bin), A2(bin), B2(bin), ...`

而探测阶段你把 A_n、B_n **两格点分组**成一个“bin-pair site”，所以探测用的单个 bin-pair 物理维度是

- 若每臂是 6D（1517-only），则 bin-pair 是 6×6 = 36D
- 原子两体是 4×4 = 16D
  这在 detection.py 里被硬检查：`dim_atom == 16`、bin_dim==6、并且效果矩阵是 36×36

------

## 1) 你代码里真正存在的“门/信道/POVM”清单 + 维度

下面表格只列**你当前 pipeline 真在跑**的$common.run_emission_to_bs 调的那条链$：

| 物理过程                                 | 代码位置/函数                                                | 作用对象                      | Schrödinger 侧算符类型                 | 单体/双体矩阵维度         |
| ---------------------------------------- | ------------------------------------------------------------ | ----------------------------- | -------------------------------------- | ------------------------- |
| 发射（原子-780纠缠）                     | `emission_gate()`；在 `run_dual_atom_emission` 里 `apply_bond_op` | `(atom 4D) ⊗ (bin 18D)`       | 幺正门                                 | **72×72**（因为 4×18=72） |
| QFC（780↔1517）                          | `qfc_gate()`；`apply_qfc()`                                  | 单个 bin(18D)                 | 幺正门                                 | **18×18**                 |
| 780 “滤除/损耗”                          | `loss_channel_both_subspaces()`；`apply_780_filter()` 固定取 K0 后选 | 单个 bin(18D)                 | Kraus 信道（目前后选）                 | 每个 Kraus **18×18**      |
| 18D→6D 投影$只保留 vac_780⊗1517$       | `project_to_1517()`                                          | 单个 bin(18D→6D)              | 线性投影/等距嵌入（当前在 state 上做） | 投影矩阵 **P: 6×18**      |
| 光纤偏振旋转（Jones）                    | `jones_gate()`；`apply_fiber_channel()`                      | 单臂单 bin（6D）              | 幺正门                                 | **6×6**                   |
| 光纤相位轮廓（phase drift/slope/jitter） | 在 `apply_fiber_channel()` 用 `jones_gate(diag(e^{iφ}))`     | 单臂单 bin（6D）              | 幺正门                                 | **6×6**                   |
| 光纤损耗（1517）                         | `loss_channel_1517_raw()`；`apply_fiber_channel()` 固定取 K0 后选 | 单臂单 bin（6D）              | Kraus 信道（目前后选）                 | 每个 Kraus **6×6**        |
| BS（两臂干涉）                           | `bs_gate_6d()`；`apply_bs()`(占位)                           | 同一 bin 的 $A_n 6D$⊗$B_n 6D$ | 两体幺正门                             | **36×36**                 |
| 探测 POVM（含效率/暗计数拆分）           | `build_detection_effects_6d()`                               | 同一 bin 的两输出端口 (6D⊗6D) | POVM effects（由 Kraus K†K 聚合）      | 每个 effect **36×36**     |
| 原子退相干（等待）                       | `_apply_atomic_dephasing()`                                  | atomA、atomB 各 4D            | Kraus 信道（采样）                     | 每个 Kraus **4×4**        |

**结论1（很关键）**：
你所谓“把门收进 POVM”，在你当前工程里，**只能对“纯光学链路”（QFC / filter / fiber / BS）做**；`emission_gate` 和原子退相干不是纯光子算符，不能和“探测 POVM”简单合并成一个纯光子 effect，否则你就把“制备过程”也挪到测量里了——那就真的换架构了。
所以闭环应该是：**TEBD 负责制备 |Ψ⟩_{atoms+bins}；POVM（可能被 Heisenberg 变换）负责把 bins 投影成点击事件。**

------

## 2) “把门收进 POVM”到底是什么：严格的等式（含维度）

先写一个你能拿去当“类型检查器”的通用恒等式：

### 2.1 单系统版本

给定状态 $\rho$（维度 d×d），测量 effect (E)（同维度 d×d），以及一个 CPTP 信道
$$
\Lambda(\rho)=\sum_\mu K_\mu \rho K_\mu^\dagger,\quad K_\mu\in\mathbb{C}^{d\times d},
$$
则 Born rule：
$$
p=\mathrm{Tr}[E,\Lambda(\rho)].
$$
把信道推到测量侧（Heisenberg/对偶映射）：
$$
p=\mathrm{Tr}[\Lambda^\dagger(E),\rho],\qquad
\Lambda^\dagger(E)=\sum_\mu K_\mu^\dagger E K_\mu.
$$
**维度检查**：

- $K_\mu^\dagger E K_\mu$：(d×d)(d×d)(d×d)→d×d
- 求和仍是 d×d
- 最终 $\mathrm{Tr}[(d×d)(d×d)]$ 合法。

若 $\Lambda$ 是幺正共轭 $\rho\mapsto U\rho U^\dagger$（U 是 d×d），那就是特例：
$$
\Lambda^\dagger(E)=U^\dagger E U.
$$

### 2.2 你真正需要的版本：原子+光子联合态

你算的是：

- 共同态 $\rho_{AP}$ 在 $\mathcal{H}_A\otimes\mathcal{H}_P$ 上
- 光学链路信道只作用在 (P) 上：$(\mathrm{Id}*A\otimes\Lambda_P)(\rho*{AP})$
- 探测只测 (P)：effect 是 $(I_A\otimes E)$

概率：
$$
p=\mathrm{Tr}\Big[(I_A\otimes E),(\mathrm{Id}*A\otimes\Lambda_P)(\rho*{AP})\Big]
=\mathrm{Tr}\Big[(I_A\otimes \Lambda_P^\dagger(E)),\rho_{AP}\Big].
$$

**你 detection.py 里算的“未归一化原子后验态”**$它用 Lüders 风格 effect 收缩出来的 spin_state$也满足同样的挪动规则：
$$
\sigma_A
=\mathrm{Tr}_P\Big[(I_A\otimes E),(\mathrm{Id}*A\otimes\Lambda_P)(\rho*{AP})\Big]
=\mathrm{Tr}*P\Big[(I_A\otimes \Lambda_P^\dagger(E)),\rho*{AP}\Big].
$$
这意味着：**在你当前“effect + 收缩”的测量假设下**，把光学链路推到 effect 上，对概率和你算出来的原子后验态是严格等价的。

------

## 3) 现在按你代码的实际顺序，把“所有该收的门”一步步收进 POVM（每步标维度）

我先只看**一个时间 bin 的两臂对**$A_n、B_n$，因为你的 BS 和探测都是逐 bin 作用的（没有跨 bin 的门），所以这是最干净的闭环单元。

### 3.0 你当前 Schrödinger 侧（简化成一个 bin-pair）的顺序

把“原子制备完毕以后”的纯光学链路抽象成：

$$
\rho^{(0)} \xrightarrow{\text{QFC}} \rho^{(1)}
\xrightarrow{\text{780 filter}} \rho^{(2)}
\xrightarrow{P_{18\to 6}} \rho^{(3)}
\xrightarrow{\text{fiber}} \rho^{(4)}
\xrightarrow{\text{BS}} \rho^{(5)}
\xrightarrow{\text{detect }E_{\rm det}} p.
$$

这里每个 $\rho^{(k)}$ 都是“原子+这个 bin-pair 的光子”的约化/局部对象；严格全局当然是张量到所有 bin，但局部维度检查在这里就够了。

------

## 3.1 从 detector 的 effect 开始：这是你代码里已有的 $E_{\rm det}$

你在 detection.py 里构造的 effect 是基于 6D×6D 的两端口空间：

- 单端口 6D（vac,H,V,2H,2V,HV）
- 两端口 36D = 6×6
  所以每个 outcome 对应一个矩阵
  $$
  E_{\rm det}(\text{pattern})\in\mathbb{C}^{36\times 36}.
  $$

------

## 3.2 把 BS 推进 effect（你最担心的那一步）

### Schrödinger：

BS 作用在同一 bin 的 $A_n 6D$⊗$B_n 6D$ 上：
$$
\rho^{(5)} = U_{\rm BS},\rho^{(4)},U_{\rm BS}^\dagger,
\quad U_{\rm BS}\in\mathbb{C}^{36\times 36}.
$$

### Heisenberg（推进到测量）：

$$
p=\mathrm{Tr}[E_{\rm det},\rho^{(5)}]
=\mathrm{Tr}[U_{\rm BS}^\dagger E_{\rm det} U_{\rm BS},\rho^{(4)}].
$$
定义
$$
E^{(4)} \equiv U_{\rm BS}^\dagger E_{\rm det} U_{\rm BS}.
$$

**维度检查**：

- $U_{\rm BS}^\dagger$ 36×36
- $E_{\rm det}$ 36×36
- $U_{\rm BS}$ 36×36
- 乘完仍 36×36 ✅

> 这一步“BS 被收进 POVM”在数学上就是 $E\leftarrow U^\dagger E U$。
> 物理上 BS 没删；实现上你可以选择“不在 state 上 apply BS”，而是在构造 effect 时做一次共轭。

------

## 3.3 把 fiber（Jones + 相位 + 损耗）推进 effect（仍然 36×36 闭环）

你代码里 fiber 是：对 A、B 两臂分别做单格点门（6×6）+ 单格点损耗 Kraus（6×6）。

我把每臂的 fiber 写成“先幺正再损耗”的 CPTP：

- 对臂 X∈{A,B}：
  - 幺正 $U_X\in\mathbb{C}^{6\times 6}$（Jones+phase 合成）
  - 损耗 Kraus ${K_{X,\mu}\subset\mathbb{C}^{6\times 6}}$$来自 `loss_channel_1517_raw`$

Schrödinger：
$$
\rho^{(4)}=(\Lambda_A\otimes\Lambda_B)(\rho^{(3)}),
\quad
\Lambda_X(\rho)=\sum_\mu K_{X,\mu}(U_X\rho U_X^\dagger)K_{X,\mu}^\dagger.
$$

Heisenberg：对偶映射是
$$
\Lambda_X^\dagger(E)=
U_X^\dagger\Big(\sum_\mu K_{X,\mu}^\dagger E K_{X,\mu}\Big)U_X.
$$

两臂一起：
$$
E^{(3)} \equiv (\Lambda_A^\dagger\otimes\Lambda_B^\dagger)\big(E^{(4)}\big).
$$

**维度检查（两臂 Kronecker 展开写清楚）**：

- 对某个 Kraus 对 $(\mu,\nu)$，定义
  $$
  K_{\mu\nu}\equiv K_{A,\mu}\otimes K_{B,\nu}\in\mathbb{C}^{36\times 36}
  $$
  因为 (6×6)⊗(6×6) = 36×36 ✅
- 先做损耗的对偶：
  $$
  \tilde E = \sum_{\mu,\nu} K_{\mu\nu}^\dagger,E^{(4)},K_{\mu\nu}
  \in\mathbb{C}^{36\times 36}
  $$
  每项都是 (36×36)(36×36)(36×36)→36×36 ✅
- 再做幺正共轭：
  $U_{AB}=U_A\otimes U_B\in\mathbb{C}^{36\times 36}$，于是
  $$
  E^{(3)} = U_{AB}^\dagger,\tilde E,U_{AB}\in\mathbb{C}^{36\times 36}.
  $$

到这里为止，你已经得到了一个**能直接作用在“6D-per-arm、未经过 fiber+BS 的 bin-pair MPS”**上的 effect。

> 也就是说：如果你愿意把“cut”放在 `project_to_1517()` 之后，那么 **完全不用引入 18D**，也不用改探测算法框架；你只要把 `effects_all`/`effects_true` 先做这串 Heisenberg 变换，再喂给现有的 run_detection_pipeline 即可。

------

## 3.4 继续把 QFC + 780 filter + 18→6 投影 也推进 effect（这一步才会出现 324×324）

你要求“所有门都收进 POVM”，那就要把 cut 再往前推到 **QFC 前**（bin 仍是 18D）。

### 3.4.1 先把 18→6 投影推进 effect

你 `project_to_1517()` 实际在 state 上做的是：
对每个 bin 施加一个线性投影矩阵 $P_{18\to 6}\in\mathbb{C}^{6\times 18}$，只保留 780=vac 的分量，然后把物理维改成 6。

对一个 bin-pair$A_n、B_n$：
$$
P_{\rm pair}\equiv P\otimes P \in \mathbb{C}^{(6\cdot 6)\times(18\cdot 18)}
=\mathbb{C}^{36\times 324}.
$$

Heisenberg：把投影推到 effect 上
$$
E^{(2)} \equiv P_{\rm pair}^\dagger,E^{(3)},P_{\rm pair}.
$$

**维度检查**：

- $P_{\rm pair}$：36×324
- $P_{\rm pair}^\dagger$：324×36
- $E^{(3)}$：36×36
  所以
  $$
  (324\times 36),(36\times 36),(36\times 324) \to 324\times 324
  $$
  ✅

于是你得到
$$
E^{(2)}\in\mathbb{C}^{324\times 324},
$$
它可以作用在“每臂 18D 的 bin-pair”上。

> 这一步解释了你前面的担心：如果你坚持全严格且 cut 放在 QFC 前，那么 effect 的局部维度确实会从 36×36 膨胀到 324×324（每个 bin-pair），计算会重很多。

------

### 3.4.2 再把 780 filter$loss_channel_both_subspaces$推进 effect

你 780 filter 的 Kraus 来自 `loss_channel_both_subspaces(eta_780=0, eta_H_1517=1, eta_V_1517=1)`，每个 Kraus 都是 18×18。
但你目前在 state 上固定取 K_list[0] 做后选。

若要“收进 POVM 且不后选”，对偶映射是：
$$
E^{(1)} \equiv \sum_{\mu,\nu}
F_{\mu\nu}^\dagger,E^{(2)},F_{\mu\nu},
\quad F_{\mu\nu}\equiv F_\mu\otimes F_\nu.
$$

**维度检查**：

- $F_\mu$ 18×18
- $F_{\mu\nu}$ 324×324
- $E^{(2)}$ 324×324
  所以每项乘法合法，结果仍 324×324 ✅

------

### 3.4.3 最后把 QFC 幺正推进 effect

QFC 是单 bin 的 18×18 幺正 $U_{\rm qfc}$。
两臂对：
$$
U_{\rm qfc,pair}\equiv U_{\rm qfc}\otimes U_{\rm qfc}\in\mathbb{C}^{324\times 324}.
$$
推进：
$$
E^{(0)} \equiv U_{\rm qfc,pair}^\dagger,E^{(1)},U_{\rm qfc,pair}\in\mathbb{C}^{324\times 324}.
$$
维度检查同 36×36 情形 ✅

------

## 3.5 最终闭环：你到底拿什么去收缩 MPS？

你 MPS 在某个 cut 处的 bin 物理维是多少，你最终就必须得到同一维度的 effect。

- 如果 cut 选在 **project_to_1517 之后**：
  - bin 单臂 6D → bin-pair 36D
  - 你要的是 $E^{(3)}\in\mathbb{C}^{36\times 36}$
  - 直接喂给 detection.py 的收缩框架（它就是 36×36 的）
- 如果 cut 选在 **QFC 之前（bin 仍 18D）**：
  - bin 单臂 18D → bin-pair 324D
  - 你要的是 $E^{(0)}\in\mathbb{C}^{324\times 324}$
  - 那么 detection.py 的整个“分组 + env 收缩”也必须支持 324D$目前它硬要求 bin_dim==6，会直接报错$

这就是你要的“编译器式闭环”：**cut 的空间维度决定了 effect 的维度**，对不上就不是等价变换，而是把物理弄丢了/重复计算了。

------

## 4) 你提到“5D 可达子空间替代 18D”：它在上面闭环里怎么插入（不破坏 BS）

这部分你之前问过，我这里把它“插进上面的维度链”说清楚：它不是魔法，只是一个合法的**子空间限制**。

### 4.1 18D 里确实有“在 BS 前永远到不了”的子空间

在你的模型里（发射门只在 780 发 0/1 光子；QFC 是数守恒的 780↔1517“类分束器”），**单臂在 BS 之前最多只有 1 个光子**。因此在单臂 18D=3×6 的基里：

- 1517 的双光子态 `2H,2V,HV`（索引 3,4,5）在 BS 前永远不该被占据；
- 同时存在“780 有光子且 1517 也有光子”的分量（总光子数 2）也不该出现。

所以 BS 前每臂真正可达的局部子空间可以取为 5D：
$$
\mathcal{H}*{\rm reach}=\mathrm{span}{
|vac\rangle,
|H*{780}\rangle,
|V_{780}\rangle,
|H_{1517}\rangle,
|V_{1517}\rangle
}
$$
（你也可以重排顺序，但维数就是 5。）

把这个 5D 嵌入 18D，用一个稀疏嵌入矩阵 $J\in\mathbb{C}^{18\times 5}$（每列就是选一个 18D 基矢），那么在 bin-pair 上
$J_{\rm pair}=J\otimes J\in\mathbb{C}^{324\times 25}$。

**于是你可以把 324×324 的 effect 压回 25×25：**
$$
E^{(0)}*{25} \equiv J*{\rm pair}^\dagger,E^{(0)},J_{\rm pair}\in\mathbb{C}^{25\times 25}.
$$
维度检查：
$$
(25\times 324)(324\times 324)(324\times 25)\to 25\times 25
$$
✅

这一步就是你说的“5D 可达子空间替代 18D”。它不会忘掉 BS，因为 BS 还在 $E^{(4)}=U_{BS}^\dagger E U_{BS}$ 那步里；你只是把**BS 之前不可达的维度**裁掉。

> 注意：**BS 后**你确实需要 6D（因为两光子会 bunching 到同一端口），所以 5D 只能用于“BS 前的每臂局部空间”。你把 BS 推进 POVM 后，state 侧就永远停留在 BS 前，因此可以一直 5D；而 6D 的“两光子端口信息”则被吸收到 effect 里了$通过 $U_{BS}^\dagger E U_{BS}$ 产生$。这并不矛盾。

------

## 5) “把算符塞进 POVM，色散/非局域信息是不是消失了？”——不会，信息只是换了载体

你这句质疑非常正确：**把门推到 POVM 并不会让复杂度凭空消失**。

- 你现在的链路（QFC、滤波、fiber、BS）都是**逐 bin（或逐 bin-pair）局域**的，所以把它们收进 POVM 仍然得到局域的 36×36（或 25×25）算符，探测收缩仍然是你 detection.py 那种“在少数 bin 上放非平凡算符，其余放 E_no”的 MPO（本质 bond dimension 1）。
- 但如果你未来加“色散/频率相关相位/跨 bin 卷积”这种会把不同时间仓耦合起来的门：
  - Schrödinger 侧：它会把 MPS 的纠缠/键维度 χ 推高；
  - Heisenberg 侧：你不在 state 上做它，就必须把它作用到 effect 上，于是 effect 会变成**跨多个 bin 的非局域 MPO**，它的 MPO bond dimension会涨。
  - 你依旧要付出多项式的收缩代价，只是复杂度从“态的 χ”转移到“算符 MPO 的 χ_op”。
    所以信息不会消失，只是从 state 的纠缠结构，搬到了 operator 的纠缠结构。

------

## 6) 你担心的“真纠缠 vs 暗计数假成功，会不会混到看不出来？”

在实验里单次 shot 本来就区分不了；你仿真也不该“凭空知道”。

但你现在的 detection.py 已经提供了**统计层面的拆分**：

- `effects_all`：含暗计数
- `effects_true`：不含暗计数（它通过 Kraus 拆分把“暗计数触发”标出来）

并且在 `compute_metrics` 时输出：

- `p_success_true_abs / p_success_false_abs`
- `fidelity_true / fidelity_false / fidelity_declared`
  这些就是回答“宣告成功里有多少是假成功、它把保真度拉低多少”的量。

如果你把损耗/QFC失败也推到 effect 上，这个“真/假拆分”并不会消失：
你只是在 `effects_all / effects_true` 生成后再做一串线性变换$$E\leftarrow U^\dagger E U$、$E\leftarrow\sum K^\dagger E K$$，线性变换不会破坏你对“暗计数分支”的标签拆分。

------

# 你要的“闭环一句话总结”

- **TEBD（MPS/TEBD）**：负责生成联合态 $\rho_{AP}$尤其是 emission_gate 那段原子-光子纠缠制备，这是不能被 POVM 取代的。
- **POVM（effect 收缩）**：负责把光子部分投影成点击模式。你完全可以把“纯光学链路”（QFC/filter/fiber/BS）通过对偶映射推到 effect 上，得到一个新的 $E_{\rm eff}$，再用同样的 MPO/MPS 收缩算概率与后验原子态。
- 每一步的维度都能像编译器一样检查：
  - 6D cut → effect 必须 36×36
  - 18D cut → effect 必须 324×324
  - 用 5D 可达子空间 → effect 可以压到 25×25
    维数对不上就说明链路没闭环。

------

如果你愿意，我下一步可以按你这条“编译器式维度闭环”标准，把你项目里**每个 gate/channel 的“adjoint 变换接口”**写成一个清单例如 `E <- conj(U_bs)`, `E <- dual_loss(K_list)`，并明确它接收/返回的矩阵维度必须是多少，这样你以后加色散/加新噪声时也能立刻检查“这一步会不会把 effect 从局域变成跨 bin 的 MPO、代价会涨到哪里”。
