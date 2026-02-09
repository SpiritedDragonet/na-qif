[![QFC as a frequency-domain linear optics with the two polarization ...](https://tse4.mm.bing.net/th/id/OIP.K2Tg8X06r_iLDbsUOBLsxAHaFQ?pid=Api)](https://www.researchgate.net/figure/QFC-as-a-frequency-domain-linear-optics-with-the-two-polarization-modes-a-Concept-of-a_fig1_320619846?utm_source=chatgpt.com)

你现在卡住的点其实是三件事混在了一起了：

1. **QFC 本体（DFG 频率转换）**到底在“量子态/算符”层面是什么操作；
2. **QFC 带来的噪声**在物理上从哪来、是不是“可以当作暗计数/背景”处理；
3. **窄带滤波（27 MHz）**既能压噪声、又会改波包形状——它为什么会导致“跨 bin 相关”，以及为什么实现时“可以不写 bin–bin 门”。

我按这三条把逻辑掰开讲清楚，并且把你代码现在做了什么、没做什么对齐一下。

------

## 0) 先对齐：你代码里 QFC 目前到底“做了什么/没做什么”

### 你现在在参数层面真正暴露出来的 QFC 旋钮很少

`common.py` 里 `QfcParams` 只有：

- `theta_H, theta_V`：两种偏振各自的“转换角”（等效频域 BS 的混合角）
- `apply_filter_780`：把**未转换的 780 分量**直接过滤掉（在模型里等价于 780 子空间损耗 η_780=0）

你目前**没有**显式参数去描述：

- QFC 的外部器件效率（57% 之类）与“纯粹的频率旋转”怎么分开；
- 1517 侧 **27 MHz 窄带滤波腔**的动力学（这是会带来跨 bin 相关的核心）；
- QFC 噪声光子的* click 概率”吸收掉）。

### 你在算符/POVM 端做的 QFC：本质上是“单 bin 的无记忆线性光学”

在 `gates.py` 的 `build_detection_effects_5d_by_bin(...)` 里，探测 POVM（effect）是这么被推回去的（Heisenberg picture）：

- 先在 1517 子空间做链路损耗通道；
- 若 `apply_filter_780`，就套一个“780 全损耗、1517 不动”的通道；
- 再用 `u_qfc = qfc_gate(theta_H, theta_V)` 做对偶映射（effect ← U† effect U）。

这一套是**每个 bin 独立**的，所以它 **不可能**产生“滤波导致的跨 bin 混合”。

### 你对 QFC 噪声的现有处理：等效为“背景计数率 → 每 bin 点击概率”

`NoiseParams` 的 `bg_rat:contentReference[oaicite:4]{index=4}hz` + `detector_gate_ns` 会先采样一个 run 级别的背景 rate，再按门宽换算成每 bin 的点击概率 `p_bg_bin_map`。代码里用的是
`p_gate = 1-exp(-rate*gate_dt)`，再用 ratio 把 gate 概念拆到 bin 上。

这等效于“背景是独立点过程（近似 Poisson）且 bin 间独立”，并且**不包含**滤波腔的时间记忆。

------

## 1) QFC 的“第一性原理”本体：为什么它等效为频域的 Beam Splitter

你现在代码里 `t实非常标准——不是拍脑袋 patch，而是来自 χ^(2) DFG 在“强泵、泵不耗尽（classical pump）”近似下的有效哈密顿量。

在 Ikuta 等人的 polarization-insensitive QFC 论文里，他们把 DFG 写成（我只描述结构不逐字抄）：

- 对于上频（780）模 (a_u) 与下频（1522/1517）模 (a_l)，有效相互作用哈密顿量形如
  (H \propto i\hbar(\xi^* a_l^\dagger a_u - \xi a_u^\dagger a_l))，耦合 (\xi) 与泵的复振幅相关。
- 解出来得到 Heisenberg 变换就是一个 **SU(2) 旋转**：输出模是输入两模的线性组合（就像 BS）。 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5962590/))

因此，在“单个时间模式/窄带近似”下，用一个 2×2（对每个偏振各一套）的旋转角 (\theta) 表示转换概率
(\eta_{\text{conv}} = \sin^2\theta)
是完全正统的。

**关键点：**
这类 traveling-wave PPLN DFG 本身是“瞬时/无记忆”的（对连续时间场来说是 (a(t)) 局域变换），所以**DFG 本体不会导致跨 bin 相关**。跨 bin 相关主要来自后面的**窄带滤波腔/窄带滤波器**（有 ringdown）。

------

## 2) QFC 这边“噪声都有什么”：从物理机制分三类

下面按“对你仿真最该关心什么”来分层。

### A. 会产生“额外光子”的噪声（最像你现在的 bg_rate）

这类噪声的共同特点：**它不是在原信号上加相位噪声，而是直接在输出波段多出随机光子**，从而制造假符合、降低 Bell 投影保真度。

典型机制（χ^(2) PPLN DFG）：

1. **自发拉曼散射（SRS，尤其 anti-Stokes Raman）**
   NIST/Stanford 的综述实验文章明确说：在实际 χ^(2) QFC 里主要噪声源之一是自发拉曼散射，并且可以通过降温降低。 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6038917/))
2. **自发参量下转换（SPDC）/参量荧光（parametric fluorescence）**
   同一篇 NIST 文章也把 χ^(2) 器件的主要噪声源点名为 SRS 和 SPDC。 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6038917/))
   （注意：是否会把 SPDC 噪声落到你的目标波段，跟泵、信号、目标三者谁最长波有关。）
3. **泵泄漏（pump leakage）**
   Fernandez-Gonzalvo 等人在 780→1552 的 QFC 实验里指出，输出附近的噪声主要可能来自泵泄漏或拉曼散射，并观察到噪声随泵功率近似线性增长。 ([arXiv](https://arxiv.org/pdf/1308.0912))

> 这三类噪声在仿真里最直接的表现，就是你现在做的：在探测端加入“背景点击率”。

------

### B. 会让“信号光子量子态变坏”的噪声（不是简单 bg_rate 能完全等效的）

这类噪声不一定产生额外光子，而是让你本来应该做的 Bell 投影因为**可区分度增加**而失败（HOM 可见度下降、纠缠保真度下降）。

典型来源：

1. **偏振两臂转换不一致（θ_H ≠ θ_V，或额外插入损耗不同）**
   Ikuta 的理论形式里 H/V 各自有独立的耦合常数与泵相位；要保持偏振量子态，需要两偏振“等效 BS 参数一致”，并且相对相位是常数可补偿。 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5962590/))
   你代码里给了 `theta_H, theta_V`，这点是对的 ，但你还缺一个“偏振相关插入损耗/相位漂移”的参数化。
2. **相位噪声 / 泵相位引入的频率转换相位**
   理论上输出里会带上泵相位（Heisenberg 解里相位显式出现）。如果这个相位在实验时间尺度上漂移并且你没有补偿/锁定，就会等效为去相干。 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5962590/))
3. **频谱/时间模式畸变（群速失配、相位匹配带宽有限）**
   这会改变波包形状、导致两节点光子不匹配，从而 HOM 降。你现在主要靠 `v_res` 这个残差可区分度旋钮兜底（代码里也写了 TODO：将来应被显式建模替代）。

------

### C. “滤波/探测系统”带来的噪声（经常跟 QFC 一起出现）

在你对标的那个 1517 nm 方案里，论文明确说：

- 780 nm 光子在 混合，通过 DFG 变成 1517 nm；
- 然后有**多级谱滤波**，包括一个 **27 MHz FWHM 的窄带滤波腔**来把强泵与 anti-Stokes Raman 背景分离；
- 在最短光纤配置时，中间站测到的背景大概 **160/170 counts per second**（两节点分别）；
- 外部器件效率（external device efficiency）约 **57%**。 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))

你代码里的默认背景均值 165 Hz、暗计数 65 Hz，跟这个量级非常贴合（bg 约 160–170 cps，SNSPD dark <65 cps）。 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/)) ## 3) “滤波对态的影响是什么？为什么影响是这样的？”

### 3.1 频域视角：滤波就是乘一个传递函数（外加真空噪声端口）

任何线性滤波器（包括 Fabry–Perot 滤波腔）对场算符在频域可以写成：

- (a_{\text{out}}(\omega)=H(\omega),a_{\text{in}}(\omega)+L(\omega),v(\omega))

其中 (v) 是损耗端口引入的真空（为了保持对易关系）。这意味着：

- 对单光子波包，频谱振幅 (f(\omega)) 被乘上 (H(\omega))：
  (f_{\text{out}}(\omega)\propto H(\omega)f(\omega))
- 若有损耗，态会变成“单光子 + 真空”的混合。

**这就解释了**为什么滤波会“改波形/改可区分度”：你把光子的频谱重新塑形了。

------

### 3.2 时域视角：窄带滤波腔必然有“记忆”（ringdown）→ 卷积

把上面做傅里叶变换，时域就是卷积：

- (a_{\text{out}}(t)=\int h(t-t'),a_{\text{in}}(t')dt' + \text{vac})的滤波腔**，(h(t)) 是指数衰减的（单极点系统）：
- (h(t)\propto e^{-(\kappa/2)t}\Theta(t))

现在把你对标实验的关键数值代进去：

- 滤波腔带宽：**27 MHz (FWHM)** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- 对应的场幅衰减时间常数（数量级）：
  (\tau_{\text{amp}}\approx 1/(\pi\Delta\nu)\approx 1/(\pi\times27,\text{MHz})\approx 11.8,\text{ns})

你的仿真常用 dt 是 0.5 ns（`dt_ns`），那 (\tau_{\text{amp}}) 就是 **二十多个 bin** 的量级（11.8/0.5≈24）。
这意味着：即使你把输入切成独立的 time-bin，滤波腔输出也会把一个 bin 的振幅“拖尾”到后面几十个 bin。

**这就是“跨 bin 相关”的真正来源：卷积（有记忆）**，不是 DFG 旋转本身。

------

## 4) “不是说影响是跨 bin 相关吗？为什么又说不用跨 bin 门？”

这句话的关键是区分：

- **现象层面：**输出 bin 之间相关（因为输出某个 bin 含有前面 bin 的成分）。
- **实现层面：**你要不要在电路里写一个直接作用在 (bin_n, bin_{n+1}) 的两体门？

答案是：**不需要直接 bin–bin 门**，因为跨 bin 相关可以由一个“记忆系统”中介产生——这在离散时间系统里等价于状态空间模型。

### 4.1 最直观类比：IIR 滤波器

经典信号处理中，一阶低通可以写成：

- (y_n = r,y_{n-1} + t,x_n)

输出 (y_n) 当然和过去输入有关（跨 bin），但实现时你只需要保存一个状态变量 (y_{n-1})，不需要把 (x_n) 和 (x_{n-1}) 直接耦合。

### 4.2 量子版：滤波腔内部模就是“隐状态/记忆”

把滤波腔内部模式记为 (a_f)，每个 time-bin 的输入模式记为 (b_n)。离散化 input–output 理论后，可以用**一串 bin 与同一个腔模顺序耦合**来实现滤波：

- 每步只做一个二体门 (U_{f,n}) 作用在 (滤波腔模 (a_f), 当前 bin (b_n)) 上；
- 因为 **同一个腔模** 会在下一步再跟 (b_{n+1}) 相互作用，所以它把记忆带过去；
- 你最后把腔模 trace 掉或测量掉，等效地就得到 bin 之间的非马尔可夫相关。

这就是我之前说的：**跨 bin 相关 ≠ 必须写 bin–bin 门**。跨 bin 相关可以通过“bin–memory 门”实现，TEBD 也更友好。

------

## 5) 回到你最关心的：QFC 噪声在仿真里应当怎么理解/建模才“更第一性原理”

### 5.1 什么时候你现在的“bg_rate → OR-map”是合理的？

如果 QFC 噪声主要是：

- anti-Stokes Raman / 泵泄漏造成的**宽带、相位随机**光子；
- 经过强滤波后仍然是**随机到达**；
- 并且这些噪声光子与信号光子在干涉上几乎不相干（不参与 HOM）

那么在“你关心的是点击模式/符合率”这个层面，把它等效为**背景计数率**是合理的工程模型。
这也正是实验论文给出来的“背景 ≈ 160–170 cps”这种标定方式 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))，跟你 `bg_rate_mean_hz` 的接口自然对齐 。

但是——它不包含滤波腔的记忆效应。

### 5.2 你现在最缺的、且“第一性原理增益最大”的两块

1. **27 MHz 滤波腔的时间记忆（跨 bin 卷积）**
   这既影响信号波形，也影响噪声的时间相关结构（尤其如果噪声在窄带里呈现 bunching/colored noise）。
2. **QFC 的“外部效率 57%”与 `theta` 的分离**
   论文里外部器件效率 57% 是“输入光纤到输出（含滤波）”的整体效率 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))。
   而 Ikuta 那种理论里 (\sin^2\theta) 描述的是“理想两模幺正混合的转换概率” ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5962590/))。
   现实里你需要一个额外的 loss channel（插入损耗、耦合损耗、滤波器透过率）把 (\eta_{\max}<1) 这件事单独拿出来，不要强行塞进 (\theta)。

------

## 6) 给你一句“落地的理解框架”（方便你以后改模型时不迷糊）

- **DFG QFC 本体：**频域 BS（SU(2) 旋转），无记忆 → 你现在的 `theta_H/theta_V` 非常正统。 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5962590/))
- **QFC 噪声光子：**主要来自泵驱动的 SRS / SPDC / 泵泄漏，往往宽带；窄带滤波能压噪声。 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6038917/))
- **27 MHz 滤波腔：*→ 自然跨 bin；实现时用“滤波腔内部模（隐状态）+ bin–腔模二体门”就能做，不必写 bin–bin 门。 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))

------

如果你愿意，我下一步可以直接把“27 MHz 滤波腔 = 一个额外记忆模”的**严格离散化公式**写成你当前工程里可用的接口（包括：你 5D bin 子空间怎么接、滤波腔模的维数要取几、以及 effect/收缩端怎么挂上去），并且把需要新增到 `common.py` 的参数列表按“与文献可对齐的字段”整理出来。





Q:当然需要，试着给一版比较第一性原理但多项式时间可以接受(最好不要超过以前的五倍到10倍)的方案

A：下面我给你一版“更第一性原理、仍然多项式时间、增量最好控制在 5–10 倍以内”的 **QFC + 1517 窄带滤波**方案；同时把你现在代码（`common.py` 里接收的参数）和文档41/相关论文里能直接读到的数值做一轮**对齐**，并把**缺失但应该加入的参数**列出来。

我会按你“刨根问底”的偏好来讲清楚三件事：

1. QFC 的物理过程与主要噪声源到底是什么、为什么会那样
2. 1517 滤波腔对“态/波包”究竟做了什么、为什么会产生跨 bin 相关
3. 怎么用“只需要 bin-辅助模二体门”的方式，把跨 bin 相关严格引入，并且复杂度仍然可控

------

## 0. 文档41里 QFC 这块到底是什么（你现在代码对应到哪里）

文档41（van Leent 等 Nature 2022）这条链路里，QFC 是把 **780 nm 单光子**用强泵浦在 PPLN 波导里做 **差频（DFG）**变成 **1517 nm**，再用 **窄带滤波腔（27 MHz）**抑制泵噪声，之后才进 0.2 dB/km 的电信光纤。文中给了几条你非常需要的“标定级”数字：

- 两个频率转换器的 **外部器件效率**：约 **57%** ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- 1517 的 **窄带滤波腔线宽**：**27 MHz (FWHM)** ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- 滤波腔 **单光子透过效率**：**81%** ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- QFC 背景（最短光纤配置下，中继站测到）：约 **160 cps** 与 **170 cps** 两路 ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- 探测器效率：> **85%**；暗计数：< **65 cps** ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- 光纤衰减：1517 约 **0.2 dB/km**（780 约 **4.0 dB/km**） ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))

你当前代码对 QFC 的实现属于：

- **“信号光”**：用 `qfc_gate()` 在 POVM 端做一个 **5×5 的单位旋转**（把 780 子空间与 1517 子空间互旋），然后 `apply_filter_780=True` 把未转换的 780 当作损耗滤掉（Heisenberg 端把 780 子空间 η=0）。这是“等效层”的实现，能得到正确的平均效率，但 **没有 1517 窄带滤波腔的记忆**。
- **“QFC 背景”**：目前用每-bin 的独立 OR-map（Poisson/伯努利）近似，把背景直接注入到探测点击概率里（仍然是 **bin 独立**）。

所以：**QFC “平均效率/平均背景率”层面你已经做了**；**缺的是 1517 滤波腔导致的时间结构/跨 bin 相关（对信号波包与噪声时间包络都相关）**。

------

## 1. QFC 噪声到底有哪些？原理是什么？

把“QFC 噪声”拆成两类，你会更清楚它在仿真里应该如何进入。

### 1.1 相干/量子态层面的“噪声”（会影响干涉可见度、贝尔投影）

这些噪声会改变**信号单光子的时间-频谱模态**，从而影响两臂在中继站的 **HOM 干涉**与 BSM 的投影纯度：

- **转换效率 < 1（幅度衰减）**：输入 780 的“单光子振幅”只有一部分被 coherently 转换成 1517，其余等效为丢失（被滤掉或散射到不可检测模式）。文档41里外部效率约 57% ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))。
- **转换引入的相位/偏振不对称**：实际系统往往用偏振多路结构（如 Sagnac/双路）保证偏振保持，但 H/V 两路的光学相位、损耗、泵功率不均会导致相对相位漂移或效率差异。你代码里允许 `theta_H/theta_V` 不同，但 **没有相位参数**（只有纯旋转）。
- **谱/时域滤波导致的模态整形**：1517 的 27 MHz 滤波腔会把信号波包做“因果卷积/记忆滤波”，这会改变到达中继站的时间模式重叠。([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))

这些属于“必须进到量子态（或者等效到非局域 POVM）里”的东西，因为它们直接影响 **BSM 投影的模式匹配**。

------

### 1.2 纯“背景光子/误点击”层面的噪声（主要制造误符合）

这些噪声往往是**非相干**的：它们不与信号保持确定相位关系，通常只通过增加 singles 与 accidentals 降低保真度：

- **泵诱导的自发噪声**：在 χ^(2) 波导里，强泵会产生宽带噪声（常见机制包括（反）斯托克斯拉曼散射、寄生参量荧光、泵光泄漏等）。你们系统为了降低噪声，选择 1607 nm 泵把输出设在 1517 nm，并且文献里提到在 1522 nm 处噪声谱密度约 41.1 photons/s/MHz，而在 1517 nm 处可降低约 8 倍（通过选泵波长/相位匹配） ([LMU Munich Theses](https://edoc.ub.uni-muenchen.de/31836/6/Leent_Tim_van.pdf))
- **滤波腔之后的残余背景计数**：文档41在中继站给了 ~160/170 cps 的 QFC 背景量级 ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- **探测器暗计数**：<65 cps ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))

这类噪声即使你仍然用 OR-map（经典叠加）处理，也能捕捉到主要的“误符合导致 fidelity 下滑”。但如果你想更严格：背景也会被 27 MHz 滤波腔赋予时间结构（尤其当噪声是窄带/热光统计时会出现 bunching），这可以用更精细模型做。

------

## 2. 1517 滤波腔对“态”的影响是什么？为什么会跨 bin？

### 2.1 连续时间的第一性原理描述：输入输出理论（Langevin/QSDE）

对一个单模滤波腔（腔模算符 (a)）与外部行波场 (b_{\rm in}(t))，标准输入输出理论给出（形式上）：

[
\dot a(t)=-(\tfrac{\kappa}{2}+i\Delta)a(t)+\sqrt{\kappa},b_{\rm in}(t)
]
[
b_{\rm out}(t)=b_{\rm in}(t)-\sqrt{\kappa},a(t)
]

这种形式在量子输入输出网络/SLH/QSDE 框架里是最基本的节点模型之一。([arXiv](https://arxiv.org/pdf/1611.00375))

这意味着：**输出场是输入场的“带记忆的线性滤波”**。把上式解出来你会得到

- 在频域：(b_{\rm out}(\omega)=H(\omega)b_{\rm in}(\omega)+\text{vac})（一个洛伦兹型传递函数）
- 在时域：(b_{\rm out}(t)) 是 (b_{\rm in}(t)) 的**因果卷积**（带指数核）

所以，如果你把场离散成 time bins（每个 bin 是一段短时间的正交模），那么：

> **第 n 个输出 bin 的算符/态，会依赖于很多个更早的输入 bin。**

这就是“跨 bin 相关”。

------

### 2.2 离散 time-bin 的严格实现：用“一个记忆模”+每步二体门（不用 bin-bin 门）

关键点：跨 bin 相关不需要你写 (H_{n,n+1}) 这种直接 bin-bin 耦合。
你只需要一个**辅助模（滤波腔记忆）**，每个时间步只做：

- **(记忆模) ⊗ (当前 time bin)** 的二体门
- 然后进入下一 bin（记忆模保留，继续作用）

离散化后，最常用的等效写法是一个“beam-splitter 型”的更新：

令第 n 个输入 bin 的湮灭算符为 (b_n)，滤波腔记忆为 (a_n)，时间步长 (\Delta t)。定义

[
r=\exp(-\kappa\Delta t/2),\quad t=\sqrt{1-r^2}
]

（(\kappa) 的定义与 FWHM 的换算我下面会给出），则理想无内部损耗的离散更新可写成：

[
a_{n+1}=r,e^{-i\Delta\Delta t},a_n + t,b_n
]
[
b^{\rm(out)}_n = t,e^{-i\Delta\Delta t},a_n + r,b_n
]

这对应一个严格的两模幺正（beam-splitter）(U_n) 在 Hilbert 空间上作用。
“跨 bin”来自于 **同一个 (a)** 不断与后续 bin 作用——相关是由记忆模传播出来的，而不是 bin-bin 直接耦合。

------

### 2.3 用文档41的 27 MHz（FWHM）估计记忆长度（帮助你选 dt 和 padding）

如果你用输入输出常见约定 (\dot a=-(\kappa/2)a+...)，那么腔的 **功率线宽 FWHM（Hz）** 与 (\kappa)（rad/s）关系近似：

[
\kappa \approx 2\pi ,{\rm FWHM}
]

于是振幅衰减时间常数（(e^{-(\kappa/2)t})）是

[
\tau_{\rm amp}\approx \frac{1}{\pi,{\rm FWHM}}
]

带入 FWHM = 27 MHz（文档41）([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))：

- (\tau_{\rm amp}\approx 1/( \pi\cdot 27,{\rm MHz})\approx 11.8,{\rm ns})
- 功率相关时间大约一半（(\sim 5.9,{\rm ns})）

这告诉你两件事：

1. 你的 bin 宽度 (\Delta t) 如果是 0.5–2 ns 这种量级，离散化是合理的。
2. 如果你只模拟 50 ns 的窗口（比如 n_bins=100, dt=0.5ns），滤波尾巴可能会撞到窗口边界；你最好在滤波之后 **加一段 vacuum padding bins** 让记忆模自然衰减干净，或者至少把窗口拉长到 (\gtrsim 5\tau)（~60 ns）以上。

------

## 3. 我给你的“比较第一性原理、仍可跑”的 QFC 方案（推荐路线）

你要的是：**严格 + 多项式时间 + 不超过原来 5–10 倍**。
我推荐你把“QFC 严格性”优先投在 **1517 滤波腔的记忆（跨 bin）**，因为它是你现在缺失且对 BSM 模式匹配最敏感的部分；至于背景噪声，先用“由谱密度+滤波带宽推导出的时间包络 Poisson”就够明显提升真实性（且成本很低）。

下面给两档方案：**A（我最推荐）** 和 **B（更通用但更重写）**。

------

## 方案 A（推荐）：在 Schrödinger 端显式加入“1517 滤波腔记忆模”，让跨 bin 相关自动出现

### A1) 核心思想

把每条臂的 1517 窄带滤波腔当作一个**小维度辅助系统**（记忆模），然后用 TEBD/两体门让它沿时间 bins 依次作用。

- **优点**：
  - 不需要把“非局域效应”硬塞进 POVM 构建（你现在的瓶颈）
  - 物理解释最直接：就是一个真实滤波腔
  - 复杂度主要增加一次“沿 bins 的扫一遍”，很容易在 5–10 倍内（尤其你本来就有 emission 这类 conveyor-belt TEBD）
- **代价**：
  - 你要在仿真里显式引入一个（或两个）新的辅助 site
  - 最后要么保留这个 site（检测时对它取 identity），要么用 padding 让它回到真空后再丢弃

------

### A2) 关键建模：记忆模维度怎么取，才能既第一性又不炸？

你的系统在“每条臂”基本是单光子发射（一次跳到 (|0\rangle/|1\rangle) 后不再激发），所以对滤波腔记忆模：

- **单条臂**：只需要截断到 **0 或 1 个光子**
- **还要保留偏振**：于是记忆模 Hilbert 空间可以取
  [
  {|{\rm vac}\rangle,\ |H\rangle,\ |V\rangle}\quad (\text{维度 }3)
  ]

两臂总记忆就是 3×3=9 维。这个非常小。

这一步非常关键：它让你“显式滤波腔”仍然是多项式、而且常数因子可控。

------

### A3) “显式 1517 滤波腔 gate”的接口设计（你要的显式 gate）

我建议你把它当作一个独立组件，接口尽量像你现在的 emission gate 那样“给参数→吐 gate”。

#### 新增参数（建议加入 `common.py`）

```python
@dataclass
class FilterCavityParams:
    fwhm_mhz: float = 27.0      # 文档41：27 MHz (FWHM)
    detuning_mhz: float = 0.0   # 滤波腔相对信号的失谐（实验会漂）
    eta_peak: float = 0.81      # 文档41：单光子透过效率 81%
    # 若想更细：区分耦合损耗 vs 插入损耗
    eta_insert: float = 1.0     # 额外插入损耗（可并入 eta_peak）
```

以及把它挂到 SimulationParams 里（两臂可各自一套）：

```python
@dataclass
class QfcAndFilterParams:
    qfc: QfcParams
    filter_a: FilterCavityParams
    filter_b: FilterCavityParams
```

#### Gate 生成函数（建议放 `physics/gates.py`）

目标：返回一个作用在
(\mathcal H_{\rm mem}(=3)\otimes \mathcal H_{\rm bin}(=5))
上的两体门 (U_{\rm filt})（维度 15×15，或 reshape 成四阶张量）。

接口建议：

```python
def filter_cavity_gate_1517(
    dt_ns: float,
    params: FilterCavityParams,
) -> np.ndarray:
    """
    Returns unitary U acting on (mem_3d ⊗ bin_5d).
    mem basis: |vac>,|H>,|V>
    bin basis: |vac>,|H780>,|V780>,|H1517>,|V1517>
    Couples ONLY the 1517 subspace to the memory, polarization-preserving.
    """
```

#### 这个 15×15 gate 的结构（你能直接照这个去写矩阵）

它是块对角的：

- 对所有涉及 **780** 的基矢：完全恒等（滤波腔不碰 780）
- 对每个偏振 (p\in{H,V})：在两维子空间
  ({|p\rangle_{\rm mem}\otimes|{\rm vac}\rangle_{\rm bin},\ |{\rm vac}\rangle_{\rm mem}\otimes|p_{1517}\rangle_{\rm bin}})
  上是一个 beam-splitter 旋转（带可选失谐相位）

典型写法（一个偏振的 2×2 块）：

[
U_p=
\begin{pmatrix}
r e^{-i\phi} & t\
-t & r e^{+i\phi}
\end{pmatrix}
]
其中
[
r=\exp(-\kappa\Delta t/2),\quad t=\sqrt{1-r^2},\quad
\phi=2\pi,\Delta,\Delta t
]

再把它嵌入到 15 维基底里即可。

> 注意：(\kappa) 用 (\kappa=2\pi,{\rm FWHM})（rad/s）是一种常用约定；你只要在代码里把“FWHM_MHz → κ”写清楚，并保证与想要的脉冲响应一致即可。这个模型与输入输出/QSDE 形式是一致的。([arXiv](https://arxiv.org/pdf/1611.00375))

#### 透过效率 0.81 怎么进？

你有两种做法：

- **做法 1（推荐，简单且稳定）**：滤波腔 gate 只负责“记忆/时间结构”（先当作无额外损耗），然后在滤波之后对 1517 子空间再施加一个本地 loss channel：(\eta_{1517}=\eta_{\rm peak}=0.81)。文档41给 81% ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- **做法 2（更第一性）**：把滤波腔做成“两端口+内损”的模型，需要每个 time step 再引入一个“损耗 bath bin”。这会让每步 gate 维度变大，成本上升更明显（但仍可能可控）。在你希望 <10× 的限制下，我建议先走做法 1。

------

### A4) 在 TEBD/矩阵乘法层面怎么操作（你问的“矩阵层面”）

你现在的 MPS/TEBD 框架已经能做“系统与 time bins 逐步作用”的 conveyor belt（你对 emitter 就这么做）。滤波腔完全同型：

**每个时间步 n：**

1. 把“滤波腔记忆 site”swap 到第 n 个 bin 的旁边
2. 对这条 bond 施加两体门 (U_{\rm filt})（就是 15×15 的矩阵 reshape 成 rank-4 张量）
3. swap 继续前进

这就是最第一性的“量子电路离散化”。

------

### A5) 复杂度为什么能控制在 5–10×？

- 记忆模维度极小：单臂 3、双臂合起来 9
- 每个时间步只增加一次（或有限次）两体门 + SVD 截断
- 对比你现在 emission 阶段已经在做的两体门+swap，这只是再多扫一遍 bins

真正决定你会不会“爆”的，是 **滤波引入的额外纠缠**会不会让 (\chi) 飙升。对线性滤波腔（被动、高斯系统）+ 单光子子空间，一般不会灾难性增长；你还可以用 `chi_max` 限住（这是可控近似）。

------

## 方案 B（更通用）：继续“把过程推到 POVM”，但把滤波腔做成 MPO/记忆通道（重写 detection contraction）

这个方案更“架构一致”（继续坚持 Heisenberg/POVM 端做通道），但你得重写一块 detection contraction，让它能收缩一个 **带 bond-dimension 的 MPO**（滤波记忆）。

- 优点：状态不变，扫参数更方便
- 缺点：实现复杂、容易引入 bug；而且记忆通道在 Heisenberg 端通常要携带 **d_mem²** 的算符空间维度（常数更大）

如果你追求的是“尽快把滤波跨 bin 做对”且不想把项目变成 MPO 编译器，我还是建议先上方案 A。

------

## 4. 把文献参数和 `common.py` 对齐（QFC/滤波/噪声相关）

下面我只对齐跟 QFC 直接相关的（你后面如果要我再把 emitter 的 Ω(t) 等也逐项对齐，我也能继续做）。

### 4.1 你代码里已有参数：推荐怎么设（含文献来源）

| `common.py` 参数                               | 物理意义                               | 文档41/文献建议值                                            | 你代码里怎么用/怎么改                                        |
| ---------------------------------------------- | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `QfcParams.theta_H`, `theta_V`                 | 780↔1517 的“相干转换角”                | 外部效率 ~57% ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/)) | 建议用：(\sin^2\theta_{H,V}=\eta_{QFC,H,V})。若取 0.57：(\theta\approx \arcsin(\sqrt{0.57})\approx 0.856) rad（比 π/4 大）。 |
| `QfcParams.apply_filter_780=True`              | 把未转换的 780 当作完全损耗            | 实验里确实会强滤除残余 780（否则光纤衰减巨大且泵噪声混入）   | 保持 True 合理。([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))（文中也强调电信传输与滤波） |
| `NoiseParams.bg_rate_mean_hz`                  | QFC 背景计数率（你现在当作等效点击率） | 中继站背景 ~160/170 cps（短纤） ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/)) | 你目前默认 165 Hz 已经对上量级。但建议拆成“谱密度×滤波带宽×链路η×探测η”，见下面缺失参数。 |
| `DetectorParams.eta_det=0.85`                  | 探测效率                               | >85% ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/)) | 默认值正好匹配。                                             |
| `NoiseParams.dark_rate_intrinsic_hz=65`        | 探测器暗计数                           | <65 cps ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/)) | 默认值正好匹配。                                             |
| `FiberChannelParams.attenuation_db_per_km=0.2` | 1517 光纤衰减                          | 0.2 dB/km ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/)) | 默认匹配。                                                   |

------

### 4.2 你代码里缺失但强烈建议加入的参数（否则“对实验指导意义”会打折）

#### (i) 1517 窄带滤波腔参数（跨 bin 的根）

- `FilterCavityParams.fwhm_mhz = 27` ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- `FilterCavityParams.eta_peak = 0.81` ([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- `FilterCavityParams.detuning_mhz`（实验中滤波腔锁定误差/漂移）
- 两臂是否同参：`filter_a`, `filter_b`（失配会直接降低 HOM 可见度）

#### (ii) QFC 噪声的“谱密度”而不只是“点击率”

你现在只用一个 `bg_rate_mean_hz`，等效上把“所有滤波/带宽/泵参数”都揉成一个数。
如果你想更第一性原理，建议至少新增：

- `qfc_noise_spectral_density_cps_per_mhz`：噪声谱密度（在目标波长附近）
  van Leent 的博士论文里给出过一个关键量级：在某配置下噪声谱密度约 41.1 photons/s/MHz（并说明通过选 1517 nm、1607 nm 泵可降低约 8 倍） ([LMU Munich Theses](https://edoc.ub.uni-muenchen.de/31836/6/Leent_Tim_van.pdf))
- 然后让背景率由
  [
  R_{\rm bg} \approx S_{\rm bg}\times \Delta\nu_{\rm filter}\times \eta_{\rm filter}\times \eta_{\rm link}\times \eta_{\rm det}
  ]
  自动生成，而不是手填 `bg_rate_mean_hz`。
  这样你改滤波腔带宽、透过率、光纤长度时，背景会自然变化。

（顺便一提：用“41.1/8 ≈ 5.1 cps/MHz”再乘 27 MHz ≈ 138 cps，和文档41的 160/170 cps 同量级，非常一致。）([LMU Munich Theses](https://edoc.ub.uni-muenchen.de/31836/6/Leent_Tim_van.pdf))

#### (iii) QFC 相位噪声/偏振不对称（影响投影纯度）

你目前 `qfc_gate()` 只有纯旋转，没有相位。建议加：

- `phi_H`, `phi_V`：转换过程给 1517 分量的相位（或等效的 Jones matrix 相位）
- 或更简单：把 QFC 视为一个 2×2 的复数“频率 beam splitter”，H/V 两路允许不同的复系数

这对指导实验的意义：你能区分“保真度掉是因为噪声误符合”还是“因为两臂模式不匹配导致投影不是纯 Bell”。

------

## 5. 最后，把“你困惑的那句”说透：跨 bin 相关到底需不需要跨 bin 门？

- **需要跨 bin 相关吗？**
  需要——只要滤波腔带宽有限，就一定是有记忆的线性系统，输出必然是输入的因果卷积（跨 bin）。([PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC9259499/))
- **需要写 bin-bin 的门吗？**
  不需要。
  你只需要把滤波腔作为一个“记忆自由度”（辅助模），每一步做“辅助模–当前 bin”的二体门。跨 bin 相关是通过辅助模的内部状态传播出来的。这个做法和你现在 emitter 的 collision model 在数学结构上是同一类（输入输出/QSDE 离散化）。([arXiv](https://arxiv.org/pdf/1902.09833))

------

## 6. 我建议你现在就落地的最小“第一性原理升级清单”（仍然可控）

如果你只做三件事，你的 QFC 仿真就会从“等效层”跳到“对实验很有指导意义”的层级：

1. **加入 1517 滤波腔记忆模（方案A）**：按上面 15×15 gate 做，沿 time bins TEBD 一遍
2. **把 QFC 背景从“手填 bg_rate_mean_hz”升级为“谱密度×带宽×链路η×探测η”**：参数可直接从论文/论文里取量级 ([LMU Munich Theses](https://edoc.ub.uni-muenchen.de/31836/6/Leent_Tim_van.pdf))
3. **给 QFC 加一个可调的 H/V 复相位参数**（哪怕先是常数相位）：让你能研究“模式失配→Bell 投影污染”这一类效应

做完这三条，你就能真正回答实验上最关键的问题之一：

> fidelity 下滑到底是“误符合主导”还是“模式不匹配主导”，以及哪个因素是你优先该去优化的（滤波锁定、泵噪声、两臂时序/频率匹配、还是探测门宽）。

------

如果你下一步希望我更“工程化”一点：我可以直接按你现有项目结构（`gates.py / detection.py / trajectory.py / mps.py` 的职责划分）给出一个**更具体的模块落点设计**，包括：

- 滤波腔 gate 的具体基底顺序（跟你 5D/6D/25D 的 embed 一致）
- TEBD 扫描时如何最少 swap（以及 padding bins 怎么选）
- 以及如何在不推翻你现有 detection engine 的前提下，把“最后多出来的滤波腔辅助 site”处理成 identity（或近似丢弃）而不引入物理错误

你只要告诉我：你最终跑仿真的典型 `dt_ns` 和 `n_bins` 大概是多少（比如是否对齐文档41的 208 ns coincidence window），我就能把“滤波腔记忆长度→需要多少 padding bins”这件事给你算得非常具体。